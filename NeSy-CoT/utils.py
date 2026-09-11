"""
NeSyKGLLM - Shared Utilities
=============================
Core functions for knowledge graph loading, symbolic rule parsing,
training data generation, model loading, evaluation, and fine-tuning.

Key design: The rule entailments are pre-computed and stored in rule_*.txt
files (CoT2 or CoT3 format). This module parses those files and uses the
instances directly as training/test data, preserving their POSITIVE/NEGATIVE
and (for CoT3) VALID/INVALID classifications.

Two CoT formats are supported:
  - CoT2 (version 1): POSITIVE/NEGATIVE classification only
  - CoT3 (version 2): [VALID]/[INVALID: ShapeName - description] +
    POSITIVE/NEGATIVE classification (v5 format with rich shape context)

Data split strategy:
  - ALL available CoT instances are pooled from rule files
  - A stratified 80:20 train/test split is applied, balanced on
    classification (CoT2) or classification × validity (CoT3)
  - No user-specified sample counts for rule-based data; sizes are
    determined by the available instances
"""

import json
import os
import re
import random
import numpy as np
import pandas as pd

from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# GPU/ML imports — guarded so data-prep-only scripts can run without them.
# ---------------------------------------------------------------------------
try:
    import torch
    import bitsandbytes as bnb
    from sklearn.metrics import (
        f1_score, accuracy_score, precision_score, recall_score,
    )
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        set_seed,
        Trainer,
        TrainingArguments,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoPeftModelForCausalLM,
    )
    _ML_AVAILABLE = True
except ImportError as e:
    _ML_AVAILABLE = False
    print(f"Note: ML libraries not fully available ({e}). "
          f"Data preparation will work, but model training/evaluation "
          f"requires torch, transformers, peft, and bitsandbytes.")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and validate the JSON configuration file.

    huggingface_token resolution order:
      1. HF_TOKEN environment variable (preferred — keeps secrets out of
         config.json, which is often shared/copied between dataset runs)
      2. cfg["huggingface_token"], if explicitly set in the file (legacy;
         avoid committing real tokens here)
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)

    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        cfg["huggingface_token"] = env_token
    elif not cfg.get("huggingface_token"):
        raise ValueError(
            "No HuggingFace token found. Set the HF_TOKEN environment "
            "variable, or (not recommended) set huggingface_token in config.json."
        )

    model_key = cfg["model_key"]
    if model_key not in cfg["available_models"]:
        raise ValueError(
            f"model_key '{model_key}' not found in available_models. "
            f"Options: {list(cfg['available_models'].keys())}"
        )
    cfg["model_name"] = cfg["available_models"][model_key]

    os.makedirs(cfg["output_dir"], exist_ok=True)
    return cfg


def set_all_seeds(seed: int):
    """Set seeds for reproducibility."""
    if _ML_AVAILABLE:
        set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def login_huggingface(token: str):
    """Login to HuggingFace hub."""
    from huggingface_hub import login
    login(token=token)
    print("Logged in to HuggingFace.")


# ---------------------------------------------------------------------------
# Knowledge Graph I/O
# ---------------------------------------------------------------------------

def preprocess_kg_file(input_file: str, output_file: str):
    """Convert tab-separated KG file to space-separated with line count header."""
    with open(input_file, "r") as f:
        lines = f.readlines()
    with open(output_file, "w") as f:
        f.write(f"{len(lines)}\n")
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                f.write(f"{parts[0]} {parts[1]} {parts[2]}\n")
    print(f"Preprocessed KG: {len(lines)} triples -> {output_file}")


def generate_relation2id(processed_kg_path: str, output_path: str):
    """Extract unique relations and write relation2id mapping."""
    relations = set()
    with open(processed_kg_path, "r") as f:
        n = int(f.readline())
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                relations.add(parts[2])
    with open(output_path, "w") as f:
        f.write(f"{len(relations)}\n")
        for i, rel in enumerate(sorted(relations)):
            f.write(f"relation_{rel}\t{rel}\n")
    print(f"Processed {n} triples with {len(relations)} unique relations -> {output_path}")


def load_entity_mapping(file_path: str) -> dict:
    """
    Load entity2id mapping file. Returns {int_id: entity_name}.

    Handles both formats:
      - Standard: name<TAB>id   (e.g. 'Cisplatin>.\\t4')
      - Reversed: id<TAB>name

    Cleans trailing '>.' suffixes and replaces underscores with spaces
    so entity names match the surface form used in CoT rule files
    (e.g. 'ALK_Positive>.' -> 'ALK Positive').
    """
    id2entity = {}
    with open(file_path, "r") as f:
        first = f.readline().strip()
        # First line is the entity count if it's a bare integer
        if not first.isdigit():
            # No count header — rewind by re-opening is simplest
            f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            # Detect format: if parts[1] is an integer, format is name\tid
            # if parts[0] is an integer, format is id\tname
            if parts[1].lstrip("-").isdigit():
                name, eid = parts[0], int(parts[1])
            elif parts[0].lstrip("-").isdigit():
                eid, name = int(parts[0]), parts[1]
            else:
                continue
            # Clean name: remove trailing '>.' artifact and normalise underscores
            name = name.rstrip(">.")
            name = name.replace("_", " ").strip()
            id2entity[eid] = name
    return id2entity


def load_knowledge_graph(file_path: str,
                         id2entity: dict = None,
                         id2relation: dict = None):
    """
    Load KG from a preprocessed triple file. Returns (graph_dict, node_list).

    The graph is keyed by REAL entity names when id2entity and id2relation
    are provided, otherwise raw string tokens from the file are used.

    Supported triple formats (auto-detected):
      - Integer IDs:  head_id tail_id relation_id   (e.g. '798 22 3')
        Requires id2entity and id2relation to decode to real names.
      - Named triples: head relation tail             (e.g. 'Germany isLeaderOf Berlin')
        Used as-is; id2entity / id2relation are ignored.

    graph[head_name][tail_name] = relation_name  (both string keys and values)
    """
    graph = {}
    nodes = set()

    def _decode(head_tok, tail_tok, rel_tok):
        """Decode a triple, applying ID maps when tokens are integers."""
        if id2entity is not None and head_tok.isdigit():
            head = id2entity.get(int(head_tok), head_tok)
        else:
            head = head_tok
        if id2entity is not None and tail_tok.isdigit():
            tail = id2entity.get(int(tail_tok), tail_tok)
        else:
            tail = tail_tok
        if id2relation is not None and rel_tok.isdigit():
            rel = id2relation.get(int(rel_tok), rel_tok)
        else:
            rel = rel_tok
        return head, tail, rel

    with open(file_path, "r") as f:
        first = f.readline().strip()
        # Skip bare integer count header (standard KGE benchmark format)
        if not first.isdigit():
            # Not a count — treat as a triple line
            parts = first.split()
            if len(parts) == 3:
                head, tail, rel = _decode(parts[0], parts[1], parts[2])
                nodes.add(head); nodes.add(tail)
                if head not in graph:
                    graph[head] = {}
                graph[head][tail] = rel

        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                head, tail, rel = _decode(parts[0], parts[1], parts[2])
                nodes.add(head); nodes.add(tail)
                if head not in graph:
                    graph[head] = {}
                graph[head][tail] = rel

    return graph, list(nodes)


def load_relation_mapping(file_path: str) -> dict:
    """
    Load relation2id mapping file. Returns {int_id: relation_name}.

    Handles both name<TAB>id and id<TAB>name column orders.
    Cleans trailing '>.' suffixes for consistency with entity names.
    """
    id2relation = {}
    with open(file_path, "r") as f:
        first = f.readline().strip()
        if not first.isdigit():
            f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            if parts[1].lstrip("-").isdigit():
                name, rid = parts[0], int(parts[1])
            elif parts[0].lstrip("-").isdigit():
                rid, name = int(parts[0]), parts[1]
            else:
                continue
            name = name.rstrip(">.")
            id2relation[rid] = name
    return id2relation


# ---------------------------------------------------------------------------
# Symbolic Rule File Parsing (Pre-generated CoTs)
# ---------------------------------------------------------------------------

def parse_rule_file(file_path: str) -> dict:
    """
    Parse a single rule .txt file (CoT2 or CoT3 format) into a structured dict.

    Auto-detects format:
      - CoT3: instances contain [VALID] or [INVALID] tags
      - CoT2: instances have no validity tags

    Returns dict with keys:
      rule_id, rule_text, head, body, instances (list of parsed instance dicts),
      pca_confidence, classification, cot_format
    """
    with open(file_path, "r") as f:
        content = f.read()

    rule_info = {
        "rule_id": None,
        "rule_text": None,
        "head": None,
        "body": None,
        "raw_instances": [],
        "instances": [],
        "pca_confidence": None,
        "classification": None,
        "cot_format": None,
    }

    # Parse rule header
    rule_match = re.search(
        r"Rule (\d+):\s*(.+?)(?=\n\nFormal Rule:)", content, re.DOTALL
    )
    if rule_match:
        rule_info["rule_id"] = rule_match.group(1)
        rule_info["rule_text"] = rule_match.group(2).strip()

    # Parse formal rule
    head_match = re.search(r"Head:\s*(.+)", content)
    body_match = re.search(r"Body:\s*(.+)", content)
    if head_match:
        rule_info["head"] = head_match.group(1).strip()
    if body_match:
        rule_info["body"] = body_match.group(1).strip()

    # Parse instances section
    instances_section = re.search(
        r"Real Instances from Knowledge Graph.*?:\n\n(.+?)(?=\n\nRule Statistics:)",
        content,
        re.DOTALL,
    )
    if instances_section:
        for line in instances_section.group(1).strip().split("\n"):
            line = line.strip()
            if line:
                rule_info["raw_instances"].append(line)

    # Parse rule statistics
    pca_match = re.search(r"PCA Confidence:\s*([\d.]+)", content)
    classification_match = re.search(r"Rule Classification:\s*(\w+)", content)
    if pca_match:
        rule_info["pca_confidence"] = float(pca_match.group(1))
    if classification_match:
        rule_info["classification"] = classification_match.group(1)

    # Auto-detect CoT format — recognises both v4 binary [VALID]/[INVALID]
    # and v5 rich [INVALID: ShapeName - description] tags
    has_validity_tags = any(
        re.search(r'\[VALID\]|\[INVALID[\]:]', inst)
        for inst in rule_info["raw_instances"]
    )
    rule_info["cot_format"] = "CoT3" if has_validity_tags else "CoT2"

    for raw_line in rule_info["raw_instances"]:
        parsed = _parse_instance_line(raw_line, rule_info["cot_format"])
        if parsed:
            rule_info["instances"].append(parsed)

    return rule_info


def _parse_instance_line(line: str, cot_format: str) -> dict:
    """
    Parse a single instance line from a rule file.

    CoT2 format:
      Entity has X, also has Y The path is classified as POSITIVE
      (PCA Confidence 0.9982 >= threshold 0.5)

    CoT3 v4 format (binary tag):
      Entity has X, also has Y. [VALID] The path is classified as POSITIVE
      (PCA Confidence 0.9273 >= threshold 0.5)

    CoT3 v5 format (rich shape tag):
      Entity has X, also has Y. [INVALID: PlayerShape - affiliation mismatch (...)]
      The path is classified as NEGATIVE (PCA Confidence 0.3100 < threshold 0.5)

    Returns dict with:
      instance_text, classification, validity, shape_name, shape_description,
      pca_confidence, pca_threshold
    """
    parsed = {
        "instance_text": line,
        "classification": None,
        "validity": None,
        "shape_name": None,        # NEW: e.g. "PlayerShape"
        "shape_description": None, # NEW: e.g. "affiliation mismatch (playsFor ≠ isAffiliatedTo)"
        "pca_confidence": None,
        "pca_threshold": None,
        "answer": None,            # Ground truth: "yes" or "no"
    }

    # Extract ground truth answer (embedded by updated NL-instances scripts)
    answer_match = re.search(r'Answer:\s*(yes|no)', line, re.IGNORECASE)
    if answer_match:
        parsed["answer"] = answer_match.group(1).lower()

    # Extract classification
    cls_match = re.search(
        r'The path is classified as (POSITIVE|NEGATIVE)', line
    )
    if cls_match:
        parsed["classification"] = cls_match.group(1)

    # Extract validity (CoT3 only) — handles both v4 and v5 tag formats
    if cot_format == "CoT3":
        # v5 rich format: [INVALID: ShapeName - description] or [INVALID: Shape1 - desc; Shape2 - desc]
        rich_invalid_match = re.search(
            r'\[INVALID:\s*([^-\]]+?)\s*-\s*([^\]]+?)\]', line
        )
        # v5 multi-shape (no description): [INVALID: Shape1; Shape2]
        multi_shape_match = re.search(
            r'\[INVALID:\s*([^\]]+)\]', line
        )
        # v4 binary format: [VALID] or [INVALID]
        binary_match = re.search(r'\[(VALID|INVALID)\]', line)

        if rich_invalid_match:
            # First shape name and description in a rich tag
            parsed["validity"] = "INVALID"
            parsed["shape_name"] = rich_invalid_match.group(1).strip()
            parsed["shape_description"] = rich_invalid_match.group(2).strip()
        elif multi_shape_match:
            # INVALID with shape name(s) but no description
            parsed["validity"] = "INVALID"
            # Take the first shape if multiple are semicolon-separated
            first_shape = multi_shape_match.group(1).split(";")[0].strip()
            parsed["shape_name"] = first_shape
        elif binary_match:
            parsed["validity"] = binary_match.group(1)

    # Extract PCA confidence and threshold
    pca_match = re.search(
        r'PCA Confidence ([\d.]+)\s*[<>=]+\s*threshold\s*([\d.]+)', line
    )
    if pca_match:
        parsed["pca_confidence"] = float(pca_match.group(1))
        parsed["pca_threshold"] = float(pca_match.group(2))

    return parsed


def load_all_rules(rules_directory: str) -> list:
    """Load all rule_*.txt files from a directory.

    Rules where no instance has an embedded 'Answer: yes/no' tag are
    skipped entirely — using them would silently fall back to the broken
    POSITIVE→yes / NEGATIVE→no heuristic and corrupt ground truth labels.
    Regenerate those rule files with the updated NL-instances scripts.
    """
    rules = []
    skipped = []
    rule_files = sorted(Path(rules_directory).glob("rule_*.txt"))
    for file_path in rule_files:
        try:
            rule_info = parse_rule_file(str(file_path))

            # --- Guard: skip rules with no Answer: tags ---
            n_with_answer = sum(
                1 for i in rule_info["instances"]
                if i.get("answer") is not None
            )
            if n_with_answer == 0 and len(rule_info["instances"]) > 0:
                skipped.append(file_path.name)
                continue

            rules.append(rule_info)
            n_inst = len(rule_info["instances"])
            n_pos = sum(1 for i in rule_info["instances"]
                        if i["classification"] == "POSITIVE")
            n_neg = sum(1 for i in rule_info["instances"]
                        if i["classification"] == "NEGATIVE")
            text_preview = (rule_info["rule_text"] or "")[:60]
            fmt = rule_info["cot_format"]
            print(
                f"  Loaded {file_path.name} [{fmt}]: {text_preview}... "
                f"({n_inst} instances: {n_pos} pos, {n_neg} neg)"
            )
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")

    if skipped:
        print(f"\n  SKIPPED {len(skipped)} rule files — no 'Answer: yes/no' tags found.")
        print(f"  Regenerate with updated NL-instances scripts:")
        for name in skipped:
            print(f"    {name}")
    print(f"\n  Total loaded: {len(rules)} rules "
          f"({len(skipped)} skipped, {len(rule_files)} total files)")
    return rules



def generate_shared_test_set(
    rules_cot3: list,
    rules_cot2: list,
    train_ratio: float = 0.8,
    include_reasoning: bool = True,
    imbalance_threshold: float = 1.5,
    max_rules_in_context: int = 3,
) -> tuple:
    """
    Generate a single shared test set from CoT3 instances, rendered in
    three formats so all variants are evaluated on identical entity paths.

    Strategy:
      1. Pool all instances from CoT3 rule files
      2. Stratified split on answer (yes/no) -> shared train keys + test keys
      3. Render test instances in three formats:
           - Baseline : path facts only          (no symbolic tags)
           - CoT2     : path facts + POSITIVE/NEGATIVE
           - CoT3     : path facts + VALID/INVALID + POSITIVE/NEGATIVE
      4. train_keys returned so prepare_data can exclude test instances from training

    Returns:
        train_keys : set of (rule_id, instance_text) for training instances
        test_keys  : set of (rule_id, instance_text) for test instances
        test_baseline_df, test_cot2_df, test_cot3_df : DataFrames for evaluation
    """
    # Per-format head-predicate indices, used to select relevant rule
    # context per test row instead of stamping one fixed global block
    # (the same static-context bug found in generate_training_data_with_rules).
    head_index_cot3 = _build_rule_head_index(rules_cot3)
    head_index_cot2 = _build_rule_head_index(rules_cot2)

    # Prefer CoT3 as the common source when both formats exist. Fall back to
    # CoT2 so a CoT2-only run can still create shared evaluation datasets.
    source_rules = rules_cot3 or rules_cot2
    source_format = "CoT3" if rules_cot3 else "CoT2"
    print(f"  Shared test source: {source_format} ({len(source_rules)} rules)")

    all_rows = []
    for rule in source_rules:
        for inst in rule["instances"]:
            all_rows.append({
                "_rule_id": rule["rule_id"],
                "_inst_key": inst["instance_text"],
                "_answer": inst.get("answer"),
                "_rule": rule,
                "_inst": inst,
            })

    if not all_rows:
        raise ValueError(
            "No CoT2 or CoT3 instances found — cannot build shared test set."
        )

    all_df = pd.DataFrame(all_rows)

    # Warn and fill if answer tag missing (old-format rule files)
    missing = all_df["_answer"].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} instances missing 'Answer: yes/no' tag. "
              f"Re-run NL-instances scripts to regenerate CoT files.")
        all_df["_answer"] = all_df["_answer"].fillna("yes")

    # Stratified split on answer (yes/no)
    train_parts, test_parts = [], []
    for key, group in all_df.groupby("_answer"):
        group = group.sample(frac=1).reset_index(drop=True)
        n_train = max(1, round(len(group) * train_ratio))
        if n_train == len(group) and len(group) >= 2:
            n_train = len(group) - 1
        train_parts.append(group.iloc[:n_train])
        test_parts.append(group.iloc[n_train:])

    train_pool = pd.concat(train_parts, ignore_index=True)
    test_pool  = pd.concat(test_parts,  ignore_index=True)

    test_keys  = set(zip(test_pool["_rule_id"],  test_pool["_inst_key"]))
    train_keys = set(zip(train_pool["_rule_id"], train_pool["_inst_key"]))

    print(f"\n  Shared test pool:  {len(test_pool)} instances "
          f"(yes={(test_pool['_answer']=='yes').sum()}, "
          f"no={(test_pool['_answer']=='no').sum()})")
    print(f"  Shared train pool: {len(train_pool)} instances")

    # --- Balance test pool if imbalanced ---
    # An imbalanced test set lets the model achieve high accuracy by always
    # predicting the majority class, collapsing Precision/Recall/F1 to 0.
    # We downsample the majority class (preferred over oversampling for test
    # sets — avoids duplicate instances inflating confidence).
    is_imbalanced, ratio, n_yes, n_no = _check_and_report_imbalance(
        test_pool, label="shared test pool", imbalance_threshold=imbalance_threshold
    )
    if is_imbalanced:
        minority_n = min(n_yes, n_no)
        yes_pool = test_pool[test_pool["_answer"] == "yes"].sample(
            n=minority_n, random_state=42
        )
        no_pool = test_pool[test_pool["_answer"] == "no"].sample(
            n=minority_n, random_state=42
        )
        test_pool = pd.concat([yes_pool, no_pool], ignore_index=True)\
            .sample(frac=1, random_state=42).reset_index(drop=True)
        # Update test_keys after downsampling
        test_keys = set(zip(test_pool["_rule_id"], test_pool["_inst_key"]))
        print(f"  Test pool balanced by downsampling majority: "
              f"{len(test_pool)} instances (yes={minority_n}, no={minority_n})")

    # Render test set in three formats using _build_cot_sample.
    # head_index selects rules relevant to each row's own rule (by shared
    # head predicate) from the appropriate rule set for that format.
    def _render_test(pool, fmt, head_index):
        samples = []
        for _, row in pool.iterrows():
            inst = row["_inst"]
            rule = row["_rule"]
            if head_index:
                relevant = _relevant_rules_for(rule, head_index, max_rules=max_rules_in_context)
                rule_ctx = create_rule_context(relevant, max_rules=max_rules_in_context)
            else:
                rule_ctx = ""
            sample = _build_cot_sample(
                inst, rule, rule_ctx, cot_format=fmt,
                include_reasoning=include_reasoning,
            )
            samples.append(sample)
        df = pd.DataFrame(samples)
        internal_cols = [c for c in df.columns if c.startswith("_")]
        return df.drop(columns=internal_cols)

    def _render_eval_test(pool):
        """
        Render the shared cross-format evaluation set.

        Every row contains:
          - Prompt  : path facts + yes/no question, NO rule context, NO tags
          - input_text : same path facts + question (for reference)
          - Label   : int 0/1 ground truth (never part of the prompt)

        This single file is used to evaluate ALL three models (Baseline,
        CoT2, CoT3) under identical conditions so their scores are directly
        comparable.
        """
        samples = []
        for _, row in pool.iterrows():
            inst = row["_inst"]
            rule = row["_rule"]

            # Re-use _build_cot_sample with Baseline format to get the
            # cleaned path_text and question — then build the eval-only prompt
            # separately so it never contains ###Response:\n<answer>.
            base_sample = _build_cot_sample(
                inst, rule, rule_context="", cot_format="Baseline",
                include_reasoning=False,
            )

            # path_text is everything in input_text before the question.
            # Recover it by stripping the question suffix from input_text.
            input_text = base_sample["input_text"]          # path_text + question
            question_part = input_text.split("Based on the rule")[0]  # everything before question
            # Safer: recompute path_text directly from cleaned instance
            path_text = _clean_instance_text(
                inst["instance_text"], cot_format="Baseline"
            ) + " "

            # Reconstruct question (same logic as _build_cot_sample)
            head_str = rule.get("head", "")
            head_match = re.match(
                r'(\?\w+)\s+(\S+)\s+(\S+)', head_str.strip()
            ) if head_str else None
            rule_text_compact = format_rule_text_compact(rule.get('rule_text', ''))
            if head_match:
                head_pred = head_match.group(2)
                question = (
                    f"Based on the rule \"{rule_text_compact}\" "
                    f"and the entailment above, "
                    f"is this {head_pred} relationship supported?"
                )
            else:
                question = (
                    f"Based on the rule \"{rule_text_compact}\" "
                    f"and the entailment above, "
                    f"is this relationship supported?"
                )

            eval_prompt = _build_eval_only_prompt(path_text, question)

            samples.append({
                "Prompt":     eval_prompt,
                "input_text": path_text + question,
                "Label":      base_sample["Label"],
            })
        return pd.DataFrame(samples)

    test_baseline_df = _render_test(test_pool, "Baseline", None)
    test_cot2_df     = _render_test(test_pool, "CoT2",     head_index_cot2)
    test_cot3_df     = _render_test(test_pool, "CoT3",     head_index_cot3)
    test_eval_df     = _render_eval_test(test_pool)

    return train_keys, test_keys, test_baseline_df, test_cot2_df, test_cot3_df, test_eval_df

def format_rule_predicate(pred: str) -> str:
    """
    Convert a raw rule predicate token into compact natural language.

    Handles FB15K-237-style path predicates ('/a/b/c/d') by keeping only
    the last two path segments -- enough to disambiguate the relation
    without repeating the full path, which was the main source of
    CoT2/CoT3 token bloat on FB15K-237 (rule_text echoed verbatim 3x
    per training row: once in the reference block, once in the question,
    once in the reasoning). Also handles YAGO-style camelCase predicates.
    """
    if not pred:
        return ''
    if '/' in pred:
        segments = [s for s in pred.strip('/').split('/') if s]
        pred = ' '.join(segments[-2:]) if len(segments) >= 2 else (segments[-1] if segments else pred)
    pred = re.sub(r'([a-z])([A-Z])', r'\1 \2', pred)   # camelCase -> spaced
    pred = pred.replace('_', ' ').replace('.', ' ')
    return re.sub(r'\s+', ' ', pred).strip().lower()


def format_rule_text_compact(rule_text: str) -> str:
    """
    Shorten a rule's natural-language text by compacting every path-like
    predicate token it contains, preserving the rule's logical structure
    (AND / then / variables) untouched.
    """
    if not rule_text:
        return rule_text
    return re.sub(r'/[\w./]+', lambda m: format_rule_predicate(m.group(0)), rule_text)


def _head_predicate(rule: dict) -> str | None:
    """Extract the head predicate token from a parsed rule's 'head' string
    (format: '?var predicate ?var_or_const')."""
    head = (rule.get("head") or "").strip().split()
    return head[1] if len(head) >= 2 else None


def _build_rule_head_index(rules: list) -> dict:
    """Group rules by head predicate for O(1) relevance lookup."""
    idx = defaultdict(list)
    for r in rules:
        hp = _head_predicate(r)
        if hp:
            idx[hp].append(r)
    return idx


def _relevant_rules_for(rule: dict, head_index: dict, max_rules: int = 3) -> list:
    """
    Select OTHER rules sharing `rule`'s head predicate, for display as
    reference context -- explicitly EXCLUDING `rule` itself.

    The current rule's own PCA confidence must NOT appear in the visible
    prompt: PCA confidence is, by construction, correlated with whether
    the rule's conclusion is actually correct, so showing the exact rule
    being asked about (with its own confidence) gives the model a direct
    numeric shortcut to the answer that bypasses reasoning over the path
    facts entirely. Confirmed empirically on FB15K-237: matched-format
    eval (context visible) hit a suspicious literal 1.0 across all
    metrics while clean eval (no context) collapsed to precision=1.0/
    recall=0.004 -- the model had learned to read the shortcut, not to
    reason, and that shortcut doesn't exist in the clean-eval format.

    The model's own generated CoT still cites a PCA confidence value as
    part of its reasoning -- but it must produce that as a learned
    association from training, not copy it out of the visible input.
    """
    hp = _head_predicate(rule)
    candidates = [r for r in head_index.get(hp, []) if r is not rule]
    return candidates[:max_rules]


def create_rule_context(rules: list, max_rules: int = 3) -> str:
    """Format rules into a context string for prompts.

    Only includes the rule text and PCA confidence value — the
    classification label is deliberately omitted to avoid leaking
    the answer into the input.
    """
    if not rules:
        return ""
    rule_context = "\n###Symbolic Rules (for reference):\n"
    for i, rule in enumerate(rules[:max_rules], 1):
        rule_context += f"{i}. {format_rule_text_compact(rule['rule_text'])}\n"
        if rule["pca_confidence"] is not None:
            rule_context += (
                f"   (PCA Confidence: {rule['pca_confidence']:.3f})\n"
            )
    rule_context += "\n"
    return rule_context


# ---------------------------------------------------------------------------
# Training Data Generation — From Pre-generated CoT Rule Files (WITH rules)
# ---------------------------------------------------------------------------

def generate_training_data_with_rules(
    graph,
    node_list,
    relation2id,
    rules,
    max_path_length=10,
    include_reasoning=True,
    use_rules=True,
    max_rules_in_context=3,
    pca_threshold=0.5,
    train_ratio=0.8,
    total_samples=None,
    negative_mining=None,
    strip_shacl_from_input=False,
    excluded_keys=None,
    imbalance_threshold=1.5,
):
    """
    Generate training data from pre-computed CoT rule instances.

    excluded_keys: optional set of (rule_id, instance_text) tuples belonging
                   to the shared test set — these are filtered out before
                   building training samples to prevent data leakage.

    When use_rules=True:
        1. Pools ALL CoT instances from rule files
        2. Stratifies on classification (POSITIVE / NEGATIVE) for both
           CoT2 and CoT3, preserving the natural VALID/INVALID
           proportions in CoT3 (no validity balancing)
        3. Within each stratum, shuffles and splits 80:20 (train_ratio)
        4. Balances POSITIVE/NEGATIVE by oversampling the minority class
        Returns (train_df, test_df)

    When use_rules=False:
        Falls back to random-walk generation (no rule entailment).
        Uses total_samples to control output size.
        Returns (train_df, test_df) via random 80:20 split.
    """
    if not use_rules:
        # Pass rules so the baseline uses rule-aligned path generation
        # (same entity pairs and head question as CoT2/CoT3, no symbolic signal).
        all_df = _generate_random_walk_data(
            graph, node_list, relation2id,
            total_samples=total_samples or 1000,
            max_path_length=max_path_length,
            include_reasoning=include_reasoning,
            rules=rules if rules else None,
            excluded_keys=excluded_keys,
        )
        # Stratified split on is_connected (Label proxy) for baseline
        strat_col = "is_connected" if "is_connected" in all_df.columns else "Label"
        train_df, test_df = _stratified_split(
            all_df, strat_col=strat_col, train_ratio=train_ratio,
        )
        return train_df, test_df

    # ----- Build from pre-generated CoT instances -----
    # Rule reference context is now computed PER RULE (not once globally):
    # each rule's instances see itself plus other rules sharing its head
    # predicate, instead of a fixed rules[:max_rules] stamped onto every row.
    head_index = _build_rule_head_index(rules)

    # Detect CoT format from the first rule that has instances
    cot_format = "CoT2"
    for rule in rules:
        if rule["instances"]:
            cot_format = rule["cot_format"]
            break
    print(f"  Detected CoT format: {cot_format}")

    # Collect ALL instances and build samples (excluding shared test instances)
    all_samples = []
    n_excluded = 0
    for rule in rules:
        relevant = _relevant_rules_for(rule, head_index, max_rules=max_rules_in_context)
        rule_context = create_rule_context(relevant, max_rules=max_rules_in_context)
        for inst in rule["instances"]:
            # Skip instances that belong to the shared test set
            if excluded_keys is not None:
                key = (rule["rule_id"], inst["instance_text"])
                if key in excluded_keys:
                    n_excluded += 1
                    continue
            sample = _build_cot_sample(
                inst, rule, rule_context, cot_format,
                include_reasoning=include_reasoning,
                strip_shacl_from_input=strip_shacl_from_input,
            )
            all_samples.append(sample)

    if n_excluded > 0:
        print(f"  Excluded {n_excluded} shared test instances from training pool.")

    if not all_samples:
        print("  WARNING: No CoT instances found. Falling back to random walk.")
        all_df = _generate_random_walk_data(
            graph, node_list, relation2id,
            total_samples=total_samples or 1000,
            max_path_length=max_path_length,
            include_reasoning=include_reasoning,
        )
        train_df, test_df = _stratified_split(
            all_df, strat_col="is_connected", train_ratio=train_ratio,
        )
        return train_df, test_df

    all_df = pd.DataFrame(all_samples)

    # --- Print pool statistics ---
    n_pos = (all_df["_classification"] == "POSITIVE").sum()
    n_neg = (all_df["_classification"] == "NEGATIVE").sum()
    n_yes = (all_df["_answer"] == "yes").sum()
    n_no  = (all_df["_answer"] == "no").sum()
    print(f"  Total CoT instances pooled: {len(all_df)}")
    print(f"    POSITIVE: {n_pos}  |  NEGATIVE: {n_neg}  (rule reliability signal)")
    print(f"    yes:      {n_yes}  |  no:       {n_no}    (ground truth answer)")

    if cot_format == "CoT3" and "_validity" in all_df.columns:
        print(f"    VALID:    {(all_df['_validity'] == 'VALID').sum()}")
        print(f"    INVALID:  {(all_df['_validity'] == 'INVALID').sum()}")

    # --- Stratification key: ground truth answer (yes/no) ---
    # This ensures both train and test have balanced yes/no distributions
    all_df["_strat_key"] = all_df["_answer"]

    # --- Stratified split ---
    train_df, test_df = _stratified_split(
        all_df, strat_col="_strat_key", train_ratio=train_ratio,
    )

    # --- Balance classes via oversampling if imbalanced ---
    train_df = _balance_classes(train_df, cot_format, imbalance_threshold)
    test_df  = _balance_classes(test_df,  cot_format, imbalance_threshold)

    # --- Final shuffle ---
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    # --- Report ---
    _print_split_report("Train", train_df, cot_format)
    _print_split_report("Test", test_df, cot_format)

    # --- Verify no leakage ---
    train_prompts = set(train_df["Prompt"])
    test_prompts = set(test_df["Prompt"])
    overlap = train_prompts & test_prompts
    if overlap:
        print(f"  WARNING: {len(overlap)} prompts appear in BOTH train and test!")
    else:
        print(f"  ✓ No prompt overlap between train and test sets.")

    # --- Compute sample weights for WeightedRandomSampler (Method 4) ---
    # Weights are attached to train_df only; test_df is always unweighted.
    # Logic:
    #   - Base weight for all samples = base_weight (default 1.0)
    #   - If enable_weighting=True AND CoT3 INVALID signal is present,
    #     NEGATIVE+INVALID samples get boosted to max_weight to emphasise
    #     hard-to-learn constraint violations.
    nm = negative_mining or {}
    if nm.get("enable", False):
        base_w  = float(nm.get("base_weight", 1.0))
        max_w   = float(nm.get("max_weight", base_w))
        do_shacl = nm.get("enable_weighting", False)

        train_df["weight"] = base_w

        if do_shacl and cot_format == "CoT3" and "_validity" in train_df.columns:
            # Boost NEGATIVE rows that also carry an INVALID constraint signal
            mask = (
                (train_df["_classification"] == "NEGATIVE") &
                (train_df["_validity"] == "INVALID")
            )
            train_df.loc[mask, "weight"] = max_w
            n_boosted = mask.sum()
            print(f"  [Weighting] base={base_w}, max={max_w}, "
                  f"boosted {n_boosted} NEGATIVE+INVALID rows")
        else:
            print(f"  [Weighting] Uniform weight={base_w} "
                  f"(SHACL boost {'disabled' if not do_shacl else 'N/A for ' + cot_format})")
    # If negative_mining is None or enable=False → no weight column → sampler skipped

    # --- Drop all internal/metadata columns — export only Prompt, input_text, output_text ---
    internal_cols = [c for c in train_df.columns if c.startswith("_")]
    train_df = train_df.drop(columns=internal_cols)
    test_df = test_df.drop(columns=internal_cols)

    return train_df, test_df


def _stratified_split(df, strat_col, train_ratio=0.8):
    """
    Split a DataFrame into train/test with proportional representation
    of each stratum.

    Each unique value of strat_col gets an 80:20 split (rounded),
    guaranteeing that even rare strata appear in both sets when possible.
    """
    train_parts = []
    test_parts = []

    for key, group in df.groupby(strat_col):
        group = group.sample(frac=1).reset_index(drop=True)  # shuffle
        n_train = max(1, round(len(group) * train_ratio))
        # Ensure at least 1 in test if group has ≥ 2 samples
        if n_train == len(group) and len(group) >= 2:
            n_train = len(group) - 1
        train_parts.append(group.iloc[:n_train])
        test_parts.append(group.iloc[n_train:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def _check_and_report_imbalance(df, label="dataset", imbalance_threshold=1.5):
    """
    Check yes/no imbalance in a DataFrame.

    Returns:
        (is_imbalanced, ratio, n_yes, n_no)
        is_imbalanced: True if majority/minority ratio exceeds imbalance_threshold
    """
    if "_answer" not in df.columns or len(df) == 0:
        return False, 1.0, 0, 0

    n_yes = (df["_answer"] == "yes").sum()
    n_no  = (df["_answer"] == "no").sum()

    if n_yes == 0 or n_no == 0:
        print(f"  WARNING [{label}]: Only one class present — yes={n_yes}, no={n_no}")
        return True, float("inf"), n_yes, n_no

    majority = max(n_yes, n_no)
    minority = min(n_yes, n_no)
    ratio = majority / minority

    is_imbalanced = ratio > imbalance_threshold
    status = f"IMBALANCED (ratio={ratio:.2f} > threshold={imbalance_threshold})" \
             if is_imbalanced else f"balanced (ratio={ratio:.2f})"
    print(f"  [{label}] yes={n_yes}, no={n_no} → {status}")

    return is_imbalanced, ratio, n_yes, n_no


def _balance_classes(df, cot_format, imbalance_threshold=1.5):
    """
    Balance the ground truth answer labels (yes/no) by oversampling
    the minority class — but ONLY if the imbalance ratio exceeds the threshold.

    imbalance_threshold: minimum majority/minority ratio to trigger balancing.
                         Default 1.5 means balancing kicks in when one class
                         is more than 1.5x the other.
    """
    if "_answer" not in df.columns or len(df) == 0:
        return df

    is_imbalanced, ratio, n_yes, n_no = _check_and_report_imbalance(
        df, imbalance_threshold=imbalance_threshold
    )

    if not is_imbalanced:
        return df  # Already balanced — no oversampling needed

    yes_df = df[df["_answer"] == "yes"]
    no_df  = df[df["_answer"] == "no"]
    target_per_class = max(len(yes_df), len(no_df))

    yes_balanced = _oversample_to(yes_df, target_per_class)
    no_balanced  = _oversample_to(no_df,  target_per_class)

    balanced = pd.concat([yes_balanced, no_balanced], ignore_index=True)
    print(f"  Balanced: yes={len(yes_balanced)}, no={len(no_balanced)} "
          f"(was yes={n_yes}, no={n_no})")
    return balanced


def _oversample_to(df, target_n):
    """Oversample a DataFrame to exactly target_n rows via repetition."""
    if len(df) == 0 or len(df) >= target_n:
        return df.copy()
    repeats = target_n // len(df)
    remainder = target_n % len(df)
    parts = [df] * repeats + [df.sample(n=remainder)]
    return pd.concat(parts, ignore_index=True)


def _print_split_report(label, df, cot_format):
    """Print distribution summary for a split."""
    print(f"\n  {label} set: {len(df)} samples")
    if "_answer" in df.columns:
        ans_counts = df["_answer"].value_counts()
        print(f"    Answer (yes/no):       {dict(ans_counts)}")
    if "_classification" in df.columns:
        cls_counts = df["_classification"].value_counts()
        print(f"    Classification (P/N):  {dict(cls_counts)}")
    if cot_format == "CoT3" and "_validity" in df.columns:
        val_counts = df["_validity"].value_counts()
        print(f"    Validity:       {dict(val_counts)}")
        # Cross-tab classification × validity
        ct = pd.crosstab(df["_classification"], df["_validity"])
        print(f"    Cross-tab (classification × validity):")
        for cls_val in ct.index:
            row = ", ".join(
                f"{v}={ct.loc[cls_val, v]}" for v in ct.columns
            )
            print(f"      {cls_val}: {row}")
        # Per-shape breakdown (v5 only — present when _shape_name is populated)
        if "_shape_name" in df.columns:
            invalid_df = df[df["_validity"] == "INVALID"]
            if not invalid_df.empty:
                shape_counts = invalid_df["_shape_name"].value_counts(dropna=False)
                print(f"    INVALID by shape:")
                for shape, count in shape_counts.items():
                    print(f"      {shape if shape else '(binary/unknown)'}: {count}")


# ---------------------------------------------------------------------------
# Sample Building (unchanged logic)
# ---------------------------------------------------------------------------

def _clean_instance_text(instance_text: str, cot_format: str,
                          strip_shacl_from_input: bool = False) -> str:
    """
    Clean the instance text for use as model input.

    Always removes:
      - The full classification sentence: "The path is classified as
        POSITIVE/NEGATIVE (PCA Confidence X.XXXX >= threshold 0.5)"
        including any dangling numeric fragments left by partial matches
      - The ground truth "Answer: yes/no" tag

    CoT3 (default): keeps [VALID]/[INVALID] bracket tags, normalises rich v5 tags
    CoT3 (strip_shacl=True): removes SHACL tags entirely (ablation mode)
    Baseline: strips all symbolic context, leaving only path facts
    """
    cleaned = instance_text

    # --- Step 1: Remove the entire classification + PCA sentence ---
    # Handles all variants:
    #   "The path is classified as POSITIVE (PCA Confidence 0.9982 >= threshold 0.5)"
    #   "The path is classified as NEGATIVE (PCA Confidence 0.3620 < threshold 0.5)"
    #   ". [VALID] The path is classified as POSITIVE (PCA Confidence 0.9273 >= threshold 0.5)"
    cleaned = re.sub(
        r'The path is classified as (?:POSITIVE|NEGATIVE)\s*'
        r'\(?(?:\s*PCA Confidence\s*[\d.]+\s*[<>=]+\s*threshold\s*[\d.]+\s*)?\)?',
        '', cleaned
    )

    # --- Step 2: Remove any remaining PCA confidence fragments ---
    # Catches orphaned fragments like "0.3620 < threshold 0.5)" or
    # "(PCA Confidence 0.9982 >= threshold 0.5)" that appear standalone
    cleaned = re.sub(
        r'\(?\s*(?:PCA Confidence\s*)?[\d.]+\s*[<>=]+\s*threshold\s*[\d.]+\s*\)?',
        '', cleaned
    )

    # --- Step 3: Handle SHACL tags based on format ---
    if cot_format == "Baseline":
        # Strip ALL symbolic context — only path facts remain
        cleaned = re.sub(r'\[(?:VALID|INVALID)[^\]]*\]', '', cleaned)
    elif strip_shacl_from_input and cot_format == "CoT3":
        # Remove ALL SHACL tags (ablation mode)
        cleaned = re.sub(r'\[(?:VALID|INVALID)[^\]]*\]', '', cleaned)
    else:
        # Normalise v5 rich tags → compact bracket form
        cleaned = re.sub(r'\[INVALID:[^\]]+\]', '[INVALID]', cleaned)
        cleaned = re.sub(r'\[VALID:[^\]]+\]', '[VALID]', cleaned)

    # --- Step 4: Remove ground truth answer tag ---
    cleaned = re.sub(r'\s*Answer:\s*(yes|no)\s*', ' ', cleaned, flags=re.IGNORECASE)

    # --- Step 5: Clean up punctuation and whitespace ---
    cleaned = re.sub(r'\.\s*\.', '.', cleaned)   # double dots
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'\s+([,.])', r'\1', cleaned)  # space before punctuation
    return cleaned


def _build_cot_sample(inst, rule, rule_context, cot_format,
                      include_reasoning=True,
                      strip_shacl_from_input=False):
    """
    Build a single training sample from a pre-generated CoT instance.

    The input contains only the cleaned entity path facts (tags and PCA
    comparison stripped by _clean_instance_text).  The output/reasoning
    contains the structured symbolic chain:
      - rule text + PCA confidence
      - CoT3 v4: "The entailed path is VALID/INVALID."
      - CoT3 v5: "The entailed path violates <ShapeName> (<description>)."
                 or "The entailed path satisfies all SHACL constraints."
      - final POSITIVE/NEGATIVE + yes/no label

    Output CSV columns: Prompt, input_text, output_text (no metadata).
    """
    instance_text = inst["instance_text"]
    classification = inst["classification"]
    validity = inst.get("validity")
    shape_name = inst.get("shape_name")
    shape_description = inst.get("shape_description")
    pca_confidence = inst.get("pca_confidence")
    rule_text = rule.get("rule_text", "")
    # Compact predicate paths (FB15K-237-style '/a/b/c/d') down to their
    # last two segments. rule_text is echoed up to 3x per row (reference
    # block, question, reasoning) so this materially reduces token count.
    rule_text_compact = format_rule_text_compact(rule_text)

    # Label: parse ground truth from "Answer: yes/no" embedded in instance text
    answer_match = re.search(r'Answer:\s*(yes|no)', instance_text, re.IGNORECASE)
    if answer_match:
        label = answer_match.group(1).lower()
    else:
        # Fallback for old-format rule files without embedded answer
        # (POSITIVE/NEGATIVE is rule reliability, not ground truth — log a warning)
        import warnings
        warnings.warn(
            f"No 'Answer: yes/no' found in instance text. "
            f"Falling back to POSITIVE→yes / NEGATIVE→no which may be incorrect. "
            f"Please regenerate CoT files with updated NL-instances scripts.",
            UserWarning, stacklevel=2
        )
        label = "yes" if classification == "POSITIVE" else "no"

    # Clean the instance text: strip classification/PCA leaks, keep path facts
    path_text = _clean_instance_text(
        instance_text, cot_format,
        strip_shacl_from_input=strip_shacl_from_input
    ) + " "

    # Build question based on the rule's head predicate
    head_str = rule.get("head", "")
    head_match = re.match(
        r'(\?\w+)\s+(\S+)\s+(\S+)', head_str.strip()
    ) if head_str else None

    if head_match:
        head_pred = head_match.group(2)
        question = (
            f"Based on the rule \"{rule_text_compact}\" and the entailment above, "
            f"is this {head_pred} relationship supported?"
        )
    else:
        question = (
            f"Based on the rule \"{rule_text_compact}\" and the entailment above, "
            f"is this relationship supported?"
        )

    # Build answer — model derives all conclusions here
    if include_reasoning:
        reasoning = f"The rule states: \"{rule_text_compact}\". "
        if pca_confidence is not None:
            reasoning += (
                f"The PCA Confidence of this rule is {pca_confidence:.4f}. "
            )
        if cot_format == "CoT3" and validity is not None:
            # Keep reasoning consistent with the [VALID]/[INVALID] tags in the input
            if validity == "INVALID":
                reasoning += "The entailed path is INVALID based on SHACL Constraints. "
            else:
                reasoning += "The entailed path is VALID based on SHACL Constraints. "
        reasoning += (
            f"The path is classified as {classification}. "
            f"The answer is {label}."
        )
        answer = reasoning
    else:
        answer = f"The answer is {label}."

    # Build full prompt
    prompt = _build_prompt(
        path_text, question, answer, rule_context,
        use_rules=True, include_reasoning=include_reasoning,
    )

    # Return columns needed for training.
    # Label (int 0/1) is kept as an explicit exported column so evaluate_model()
    # always has a reliable ground truth regardless of output_text parsing.
    # It does NOT start with "_" so it survives the internal-column purge.
    sample = {
        "Prompt": prompt,
        "input_text": path_text + question,
        "output_text": answer,
        "Label": 1 if label == "yes" else 0,
    }

    # Keep classification and answer internally for stratified splitting (dropped before CSV export)
    sample["_classification"] = classification
    sample["_answer"] = label   # ground truth yes/no — used for stratification and balancing
    if cot_format == "CoT3":
        sample["_validity"] = validity
        sample["_shape_name"] = shape_name  # kept for diagnostics, dropped before export

    return sample


def _build_eval_only_prompt(path_text: str, question: str) -> str:
    """
    Build the shared cross-format evaluation prompt.

    Contains ONLY:
      - The ###Instruction header (identical for all three variants)
      - path facts (entity names, relations — no [VALID]/[INVALID] tags,
        no POSITIVE/NEGATIVE, no PCA confidence, no rule context)
      - the yes/no question
      - the ###Response: marker (model generates from here)

    This is the single prompt format used to evaluate ALL three fine-tuned
    models (Baseline, CoT2, CoT3) fairly — none of them see any symbolic
    signal at inference time.

    The answer field is intentionally omitted so that when this prompt is
    stored in the CSV the model never sees the ground truth.  The Label
    column carries the ground truth for the evaluation script.
    """
    return (
        f"###Instruction:\nAnswer the following yes/no question by "
        f"reasoning step-by-step.\n\n"
        f"###Input:\n{path_text}{question}\n\n"
        f"###Response:"
    )


def _build_prompt(path_text, question, answer, rule_context,
                  use_rules=True, include_reasoning=True):
    """Build the formatted prompt string."""
    if include_reasoning:
        if use_rules and rule_context:
            prompt = (
                f"###Instruction:\nAnswer the following yes/no question by "
                f"reasoning step-by-step. Use the symbolic rules as additional "
                f"context along with the path information.\n{rule_context}"
                f"###Input:\n{path_text}{question}\n\n###Response:\n{answer}"
            )
        else:
            prompt = (
                f"###Instruction:\nAnswer the following yes/no question by "
                f"reasoning step-by-step.\n\n###Input:\n{path_text}{question}"
                f"\n\n###Response:\n{answer}"
            )
    else:
        if use_rules and rule_context:
            prompt = (
                f"{rule_context}###Input:\n{path_text}{question}"
                f"\n\n###Response:\n{answer}"
            )
        else:
            prompt = (
                f"###Input:\n{path_text}{question}\n\n###Response:\n{answer}"
            )
    return prompt


# ---------------------------------------------------------------------------
# Training Data Generation — Baseline (WITHOUT rules, same task as CoT2/CoT3)
# ---------------------------------------------------------------------------

def _generate_random_walk_data(
    graph,
    node_list,
    relation2id,
    total_samples=1000,
    max_path_length=10,
    include_reasoning=True,
    rules=None,
    excluded_keys=None,
):
    """
    KG-LLM baseline data generation via DFS random walks over real KG triples.

    Always delegates to _dfs_baseline_walk. The `rules` and `excluded_keys`
    parameters are accepted for API compatibility with generate_training_data_with_rules
    but are not used — the DFS walk samples independently from the full KG and
    therefore cannot replay shared test instances by construction.
    """
    return _dfs_baseline_walk(
        graph, node_list, relation2id,
        total_samples=total_samples,
        max_path_length=max_path_length,
        include_reasoning=include_reasoning,
    )


def _dfs_baseline_walk(
    graph,
    node_list,
    relation2id,
    total_samples=1000,
    max_path_length=10,
    include_reasoning=True,
):
    """
    KG-LLM baseline: DFS random walks over real KG triples.

    Faithful to the KG-LLM paper (Luo et al., 2024):
      - Walks use REAL entity names and REAL relation names from the KG
        (e.g. "Germany has isLeaderOf with Angela_Merkel"), not anonymised
        node_Q35666 identifiers.
      - The walk is DFS with backtracking — from the start entity, follow a
        random outgoing edge, recurse, avoid revisiting nodes.
      - The final question asks about the specific relation on the LAST edge
        of the path (the "target" triple to predict), matching the task
        format of CoT2/CoT3: "Does <head> have <relation> with <tail>?"
      - Ground truth (Label=1) is whether that triple actually exists in the
        KG — positive paths end on a real KG edge, negative paths do not.
      - No rule context, no POSITIVE/NEGATIVE label, no VALID/INVALID tag,
        no PCA confidence — the model must reason from path facts alone.

    This ensures Baseline, CoT2, and CoT3 all train and are evaluated on
    structurally identical prompts; the only variable is the symbolic signal
    present during training.
    """
    data = []
    unique_paths: set = set()
    pos_count = 0
    neg_count = 0
    target_per_class = total_samples // 2

    # graph[head][tail] = relation_name  (all strings — decoded by load_knowledge_graph)
    # Build flat edge list for sampling
    all_edges = [
        (h, t, rel)
        for h, neighbors in graph.items()
        for t, rel in neighbors.items()
    ]

    if not all_edges:
        print("  ERROR: Graph has no edges — cannot generate baseline samples.")
        return pd.DataFrame()

    max_attempts = total_samples * 100
    attempts = 0

    while len(data) < total_samples and attempts < max_attempts:
        attempts += 1

        want_positive = pos_count < target_per_class
        want_negative = neg_count < target_per_class
        if not want_positive and not want_negative:
            break

        # Sample a real edge as the anchor triple to predict
        head, true_tail, rel_name = random.choice(all_edges)

        if want_positive and (not want_negative or random.random() < 0.5):
            target_tail = true_tail
            is_positive = True
        else:
            # Corrupt tail: pick any entity not already a neighbour of head
            neighbours = set(graph.get(head, {}).keys())
            candidates = [n for n in node_list if n not in neighbours and n != head]
            if not candidates:
                continue
            target_tail = random.choice(candidates)
            is_positive = False

        # DFS walk from head over OTHER edges (target endpoints excluded)
        path_length = random.randint(1, max_path_length - 1)
        visited = {head, target_tail}
        path_steps = []   # list of (from_entity, relation_name, to_entity)
        current = head

        for _ in range(path_length):
            if current not in graph:
                break
            neighbours_here = [
                (nbr, rel)
                for nbr, rel in graph[current].items()
                if nbr not in visited
            ]
            if not neighbours_here:
                break
            next_node, next_rel = random.choice(neighbours_here)
            path_steps.append((current, next_rel, next_node))
            visited.add(next_node)
            current = next_node

        if not path_steps:
            continue

        # Render path: "EntityA has relationX with EntityB."
        path_text = ""
        reasoning_text = ""
        for (src, rel, tgt) in path_steps:
            step_str = f"{src} has {rel} with {tgt}. "
            path_text += step_str
            if include_reasoning:
                reasoning_text += step_str

        question = f"Does {head} have {rel_name} with {target_tail}?"
        dedup_key = path_text + question
        if dedup_key in unique_paths:
            continue
        unique_paths.add(dedup_key)

        label_str = "yes" if is_positive else "no"
        answer = (reasoning_text if include_reasoning else "") + \
                 f"The answer is {label_str}."

        prompt = _build_prompt(
            path_text, question, answer, rule_context="",
            use_rules=False, include_reasoning=include_reasoning,
        )

        data.append({
            "Prompt":       prompt,
            "input_text":   path_text + question,
            "output_text":  answer,
            "Label":        1 if is_positive else 0,
            "has_rule_context": False,
            "is_connected": is_positive,
        })

        if is_positive:
            pos_count += 1
        else:
            neg_count += 1

        if len(data) % 200 == 0:
            print(f"  Generated {len(data)}/{total_samples} baseline samples...")

    if len(data) < total_samples:
        print(
            f"  WARNING: Only generated {len(data)}/{total_samples} baseline "
            f"samples after {attempts} attempts."
        )

    print(
        f"  Generated {len(data)} DFS baseline samples "
        f"(positive: {pos_count}, negative: {neg_count})"
    )
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Model Utilities (require GPU libraries)
# ---------------------------------------------------------------------------

def _check_ml_available():
    """Raise a clear error if ML libraries weren't loaded."""
    if not _ML_AVAILABLE:
        raise ImportError(
            "GPU/ML libraries (torch, transformers, peft, bitsandbytes) "
            "are not available. This is likely due to a missing system "
            "library (e.g. GLIBCXX_3.4.29). These are required for "
            "model training and evaluation but NOT for data preparation."
        )


def create_bnb_config(cfg: dict) -> "BitsAndBytesConfig":
    """Create BitsAndBytes quantization config from the JSON config."""
    _check_ml_available()
    q = cfg.get("quantization", {})
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    compute_dtype = dtype_map.get(
        q.get("bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16
    )
    return BitsAndBytesConfig(
        load_in_4bit=q.get("load_in_4bit", True),
        bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model(model_name: str, bnb_config, max_memory_mb: int = 40960):
    """Load a quantized model and tokenizer.

    max_memory is specified for BOTH GPU and CPU to prevent device_map="auto"
    from silently offloading unlimited model layers to CPU RAM (which causes
    the OOM kill seen in SLURM jobs that die within 2 minutes of starting).

    CPU RAM cap: 24GB — enough headroom for the OS, data loaders, and
    tokenizer while preventing runaway offloading.
    """
    _check_ml_available()
    n_gpus = torch.cuda.device_count()
    print(f"  Detected {n_gpus} GPU(s). GPU memory cap: {max_memory_mb}MB each.")

    if n_gpus == 0:
        raise RuntimeError(
            "No GPUs detected by torch.cuda.device_count(). "
            "Check that the SLURM job has --gres=gpu:1 and that CUDA is "
            "accessible in the conda environment."
        )

    gpu_max = f"{max_memory_mb}MB"
    cpu_max = "24GiB"   # hard cap on CPU RAM offloading

    max_memory = {i: gpu_max for i in range(n_gpus)}
    max_memory["cpu"] = cpu_max

    print(f"  max_memory: GPU={gpu_max}, CPU={cpu_max}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
    )

    # Report where each layer landed
    if hasattr(model, "hf_device_map"):
        devices = {}
        for layer, dev in model.hf_device_map.items():
            devices[str(dev)] = devices.get(str(dev), 0) + 1
        print(f"  Layer placement: {devices}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Truncate from the LEFT (oldest context first) so that if a prompt
    # still exceeds max_length after the token-bloat fixes below, the
    # loss ###Response:...The answer is X. (always last) is never clipped.
    tokenizer.truncation_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def find_all_linear_names(model) -> list:
    """Find all Linear4bit module names for LoRA targeting."""
    _check_ml_available()
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(
                names[0] if len(names) == 1 else names[-1]
            )
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def preprocess_batch(batch, tokenizer, max_length=512):
    """
    Tokenize a batch of prompts.

    padding=False here (was "max_length"): fine_tune_model() already uses
    DataCollatorForLanguageModeling, which pads dynamically to the longest
    sequence in each actual training batch. Pre-padding every row to
    max_length here was wasted compute -- every batch paid for the full
    max_length regardless of how long its rows actually were.
    """
    return tokenizer(
        batch["Prompt"], truncation=True, max_length=max_length,
        padding=False,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model, tokenizer, test_df, max_samples=200, max_new_tokens=50,
    device="cuda", log_path=None,
    repetition_penalty=1.15, no_repeat_ngram_size=4,
    max_input_tokens=None,
) -> dict:
    """
    Evaluate a model on the test set. Returns metrics dict.

    Ground truth resolution (in priority order):
      1. Label column (int 0/1) — always present in CSVs generated by this
         pipeline. This is the canonical, reliable source.
      2. Regex on output_text for "The answer is yes/no" — fallback for
         legacy CSVs that predate the Label column.

    If neither source is available for a row, it is skipped and a warning
    is printed so the caller can regenerate the data.

    Truncation guard: prompts longer than the tokenizer's model_max_length
    are truncated on the LEFT (keeping the ###Response: suffix) so the
    model always generates from the correct position. A count of truncated
    prompts is reported at the end.

    log_path: if set, writes a per-row CSV with the raw generation text and
    how the yes/no prediction was resolved (explicit / bare_word / tiebreak /
    tiebreak_default_no). Essential for diagnosing metrics that look
    suspicious (e.g. exactly 1.0, or a collapse to one class) without
    guessing from aggregate statistics -- inspect what the model actually
    generated instead of inferring it indirectly.
    """
    _check_ml_available()
    model.eval()
    y_true = []
    y_pred = []
    raw_records = [] if log_path else None
    resolution_counts = {"explicit": 0, "bare_word": 0, "tiebreak": 0, "tiebreak_default_no": 0}

    n = min(len(test_df), max_samples)
    print(f"Evaluating on {n} samples...")

    # Check whether Label column is present and warn once if not
    has_label_col = "Label" in test_df.columns
    if not has_label_col:
        print(
            "  WARNING: 'Label' column not found in test CSV. "
            "Falling back to parsing output_text for ground truth. "
            "Regenerate CSVs with the updated pipeline to fix this."
        )

    n_fallback = 0       # rows where Label was absent → fell back to regex
    n_skipped  = 0       # rows where neither source gave a valid label
    n_truncated = 0      # prompts that exceeded max_length and were truncated

    # Resolve tokenizer max length once (guards against very long YAGO entity names)
    tok_max = getattr(tokenizer, "model_max_length", 2048)
    # Some tokenisers report absurdly large values (e.g. 1e30); cap sensibly
    if tok_max > 8192:
        tok_max = 2048
    if max_input_tokens is not None:
        if max_input_tokens <= 0:
            raise ValueError("evaluation.max_input_tokens must be positive")
        tok_max = max_input_tokens
    model_context = getattr(model.config, "max_position_embeddings", None)
    if isinstance(model_context, int) and model_context > 0:
        tok_max = min(tok_max, model_context - max_new_tokens)
    if tok_max <= 0:
        raise ValueError("Generation budget leaves no room for the input prompt")

    for idx, row in test_df.head(n).iterrows():

        # ------------------------------------------------------------------
        # 1. Build eval prompt (everything up to and including ###Response:)
        # ------------------------------------------------------------------
        prompt = row["Prompt"]
        if "###Response:" in prompt:
            eval_prompt = prompt[:prompt.index("###Response:") + len("###Response:")]
        else:
            eval_prompt = prompt

        # ------------------------------------------------------------------
        # 2. Resolve ground truth
        # ------------------------------------------------------------------
        if has_label_col and pd.notna(row.get("Label")):
            expected_yes = int(row["Label"]) == 1
        else:
            # Fallback: parse "The answer is yes/no" from output_text
            gt_match = re.search(
                r'The answer is (yes|no)', str(row.get("output_text", "")),
                re.IGNORECASE
            )
            if gt_match:
                expected_yes = gt_match.group(1).lower() == "yes"
                n_fallback += 1
            else:
                # Cannot determine ground truth — skip this row
                n_skipped += 1
                continue

        # ------------------------------------------------------------------
        # 3. Tokenise with truncation guard
        #    Truncate from the LEFT so ###Response: is always at the end.
        # ------------------------------------------------------------------
        inputs = tokenizer(
            eval_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=tok_max,
        ).to(device)

        # Detect if truncation occurred (tokenised length == tok_max)
        if inputs["input_ids"].shape[1] == tok_max:
            n_truncated += 1

        input_len = inputs["input_ids"].shape[1]

        # ------------------------------------------------------------------
        # 4. Generate only new tokens
        # ------------------------------------------------------------------
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                # Pure greedy decoding with no repetition guard was found to
                # loop indefinitely on out-of-distribution (clean-eval) inputs
                # -- e.g. hallucinating a chain of unrelated "The rule
                # states..." continuations and never reaching "the answer is
                # X" before max_new_tokens ran out (confirmed via raw
                # generation logs: 70% of clean-eval CoT2 rows fell through
                # to the tiebreak_default_no fallback for exactly this
                # reason). repetition_penalty discourages exact token loops;
                # no_repeat_ngram_size hard-blocks repeating the same 3-token
                # sequence, which is enough to break "then X role?b.\" The
                # rule states"-style cycles without materially changing
                # legitimate short answers like "The answer is no."
                # Pure greedy decoding with no repetition guard was found to
                # loop indefinitely on out-of-distribution (clean-eval) inputs
                # (confirmed via raw generation logs). repetition_penalty=1.3
                # + no_repeat_ngram_size=3 fixed the infinite loop but was
                # too aggressive for this 1B model -- it started producing
                # garbled/malformed tokens (stray non-ASCII characters,
                # erratic capitalization, fused words) instead, trading one
                # failure mode for a milder one. Defaults here are toned
                # down; sweep both if generations still look degraded.
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

        # Decode ONLY newly generated tokens (avoids 'no' in entity names
        # inside the prompt being matched by the extraction regex)
        new_tokens = outputs[0][input_len:]
        model_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # ------------------------------------------------------------------
        # 5. Extract yes/no from model output
        #    Priority: "The answer is X" pattern → word-boundary match → last occurrence
        # ------------------------------------------------------------------
        ans_match = re.search(r'the answer is (yes|no)', model_answer.lower())
        if ans_match:
            model_has_yes = ans_match.group(1) == "yes"
            resolution_type = "explicit"
        else:
            has_yes = bool(re.search(r'\byes\b', model_answer.lower()))
            has_no  = bool(re.search(r'\bno\b',  model_answer.lower()))
            if has_yes and not has_no:
                model_has_yes = True
                resolution_type = "bare_word"
            elif has_no and not has_yes:
                model_has_yes = False
                resolution_type = "bare_word"
            else:
                # Ambiguous or empty — use last occurrence as tiebreaker
                last_yes = model_answer.lower().rfind("yes")
                last_no  = model_answer.lower().rfind("no")
                model_has_yes = last_yes > last_no
                resolution_type = "tiebreak_default_no" if (last_yes == -1 and last_no == -1) else "tiebreak"

        resolution_counts[resolution_type] += 1

        if raw_records is not None:
            raw_records.append({
                "idx": idx,
                "expected_yes": expected_yes,
                "predicted_yes": model_has_yes,
                "correct": expected_yes == model_has_yes,
                "resolution_type": resolution_type,
                "prompt_tail": eval_prompt[-500:],   # last 500 chars for context, not the full (often huge) rule-context prompt
                "raw_generation": model_answer,
            })

        y_true.append(expected_yes)
        y_pred.append(model_has_yes)

        if (len(y_true)) % 50 == 0:
            print(f"  Processed {len(y_true)}/{n} samples...")

    # ------------------------------------------------------------------
    # 6. Diagnostics before computing metrics
    # ------------------------------------------------------------------
    n_eval = len(y_true)
    pred_yes = sum(y_pred)
    pred_no  = n_eval - pred_yes
    true_yes = sum(y_true)
    true_no  = n_eval - true_yes

    print(f"\n  Evaluation summary ({n_eval} samples evaluated):")
    print(f"    Ground truth  — yes: {true_yes}, no: {true_no}")
    print(f"    Predictions   — yes: {pred_yes}, no: {pred_no}")
    print(f"    Resolution types — explicit: {resolution_counts['explicit']}, "
          f"bare_word: {resolution_counts['bare_word']}, "
          f"tiebreak: {resolution_counts['tiebreak']}, "
          f"tiebreak_default_no: {resolution_counts['tiebreak_default_no']}")
    if resolution_counts['tiebreak_default_no'] > n_eval * 0.1:
        print(f"    WARNING: {resolution_counts['tiebreak_default_no']}/{n_eval} rows fell through "
              f"to the tiebreak_default_no fallback (model generated neither 'yes' nor 'no'). "
              f"Metrics may reflect generation failure, not model reasoning — inspect raw_generation "
              f"in the log CSV.")
    if n_fallback:
        print(f"    Fallback label resolution (no Label col): {n_fallback} rows")
    if n_skipped:
        print(f"    WARNING: Skipped {n_skipped} rows (no resolvable ground truth)")
    if n_truncated:
        print(
            f"    WARNING: {n_truncated}/{n_eval} prompts were truncated to "
            f"{tok_max} tokens. Consider increasing evaluation.max_input_tokens in config.json."
        )

    if raw_records is not None:
        pd.DataFrame(raw_records).to_csv(log_path, index=False)
        print(f"    Raw generation log written to: {log_path}")

    if n_eval == 0:
        print("  ERROR: No samples could be evaluated. Check Label column and output_text format.")
        return {
            "accuracy": 0.0, "f1_score": 0.0,
            "precision": 0.0, "recall": 0.0,
            "y_true": [], "y_pred": [],
        }

    accuracy  = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, pos_label=True, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=True, zero_division=0)
    recall    = recall_score(y_true, y_pred, pos_label=True, zero_division=0)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "y_true": y_true,
        "y_pred": y_pred,
        "resolution_counts": resolution_counts,
    }


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def fine_tune_model(model, tokenizer, train_dataset, output_dir: str,
                    cfg: dict, sampler=None):
    """Fine-tune a model with LoRA + QLoRA. Returns the fine-tuned model.

    Args:
        sampler: Optional WeightedRandomSampler (Method 4). Injected via
                 _get_train_sampler so the standard HF DataLoader and collator
                 are completely untouched — only sampling order changes.
    """
    _check_ml_available()
    tcfg = cfg.get("training", {})
    lcfg = cfg.get("lora", {})

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    modules = find_all_linear_names(model)
    peft_config = LoraConfig(
        r=lcfg.get("r", 16),
        lora_alpha=lcfg.get("lora_alpha", 64),
        target_modules=modules,
        lora_dropout=lcfg.get("lora_dropout", 0.1),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)"
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 4),
        warmup_steps=tcfg.get("warmup_steps", 10),
        max_steps=tcfg.get("num_steps", 500),
        learning_rate=tcfg.get("learning_rate", 2e-4),
        fp16=tcfg.get("fp16", True),
        logging_steps=tcfg.get("logging_steps", 50),
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=tcfg.get("save_steps", 100),
    )

    if sampler is not None:
        _sampler = sampler

        class _WeightedTrainer(Trainer):
            # HF passes `dataset` as a positional arg in newer versions;
            # accept it but ignore it — we always return the fixed sampler.
            def _get_train_sampler(self, dataset=None):
                return _sampler

        print("WeightedRandomSampler active (Method 4)")
        trainer = _WeightedTrainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
    else:
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

    model.config.use_cache = False
    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    return model


def print_results(label: str, results: dict):
    """Pretty-print evaluation results."""
    print(f"\n{label}")
    print(f"   Accuracy:  {results['accuracy']:.3f}")
    print(f"   F1 Score:  {results['f1_score']:.3f}")
    print(f"   Precision: {results['precision']:.3f}")
    print(f"   Recall:    {results['recall']:.3f}")


def save_results_json(path: str, results: dict):
    """Save results dict to JSON."""
    clean = {}
    for k, v in results.items():
        if isinstance(v, dict):
            clean[k] = {
                kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                for kk, vv in v.items()
                if kk not in ("y_true", "y_pred")
            }
        elif isinstance(v, (float, np.floating)):
            clean[k] = float(v)
        else:
            clean[k] = v
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"Results saved to {path}")

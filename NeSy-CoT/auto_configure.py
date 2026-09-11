#!/usr/bin/env python3
"""
NeSyKGLLM - Config Auto-Tuner / Pre-Flight Diagnostic
========================================================
Measures dataset-specific properties that config.json currently hardcodes
as fixed constants copied across datasets, and writes a dataset-specific
config with measured values instead of guessed ones.

WHY THIS EXISTS
---------------
config.json is a static file -- it cannot compute anything. Reusing the
same file across KGs (swapping only data_dir / rules_dir / output_dir)
silently carries forward values tuned for one dataset's structure onto
another with very different properties. This is what happened with
FB15K-237 inheriting FrenchRoyalty-tuned max_length=512 and
max_new_tokens=50: FrenchRoyalty's short entity/relation names never
needed more, FB15K-237's long path-style predicates did, and the
mismatch silently truncated training answers and eval generations.

WHAT IT MEASURES
-----------------
  training.max_length          <- Prompt token length distribution
                                   (99th percentile, rounded up)
  evaluation.max_new_tokens    <- output_text token length distribution
                                   (99th percentile + margin)
  training.num_steps           <- derived from train pool size and a
                                   target epoch count (default 3), so a
                                   small KG doesn't get massively
                                   over-fit and a large one isn't
                                   under-trained by a flat step count
  data_generation.pca_threshold <- derived from the mined rules' PCA
                                   confidence distribution (median), so
                                   the POSITIVE/NEGATIVE split isn't
                                   collapsed to one class on a KG whose
                                   rules score very differently than
                                   whatever dataset 0.5 was tuned on
  data_generation.max_path_length <- derived from the mined rules' body
                                   atom counts (max + small margin)

WHAT IT DELIBERATELY DOES NOT AUTO-SET
----------------------------------------
  relation_skew_threshold -- this needs the iterative
      filter_skewed_relations() diagnostic (prepare_data.py) run against
      several candidate thresholds and a judgment call about how much
      data loss is acceptable; not a single-pass statistic.
  learning_rate / lora config / imbalance_threshold -- these are general
      training hygiene knobs, not KG-structure-dependent (see prior
      discussion on which config keys should vary per dataset).

USAGE
-----
  # Run AFTER prepare_data.py has generated train_data_with_rules_CoT2.csv
  # for the target dataset (needed to measure token lengths).
  python auto_configure.py \\
      --config config.json \\
      --cot_version CoT2 \\
      --output config_FB15K-237.json

  # Dry run -- print measurements without writing a file
  python auto_configure.py --config config.json --cot_version CoT2 --dry_run
"""

import argparse
import copy
import json
import math
import os
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_tokenizer(model_name: str, hf_token):
    """
    Load the real tokenizer for accurate measurements. Falls back to the
    ungated SmolLM2 substitute if the gated model isn't accessible --
    but WARNS loudly, because token counts differ between tokenizers and
    a fallback measurement should not be trusted for the final max_length
    decision, only as a rough sanity check.
    """
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        print(f"  Using real tokenizer: {model_name}")
        return tok, True
    except Exception as e:
        print(f"  WARNING: Could not load '{model_name}' tokenizer ({e}).")
        print(f"  Falling back to HuggingFaceTB/SmolLM2-135M-Instruct for a "
              f"ROUGH estimate only -- token counts will not exactly match "
              f"the real model. Re-run with HF access for final numbers.")
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
        return tok, False


def round_up(value: float, base: int) -> int:
    """Round up to the nearest multiple of `base` (e.g. 128, 50)."""
    return int(math.ceil(value / base) * base)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def measure_prompt_lengths(train_csv: str, tokenizer) -> dict:
    """Token length distribution of the full training Prompt column."""
    df = pd.read_csv(train_csv)
    if "Prompt" not in df.columns:
        raise ValueError(f"'Prompt' column not found in {train_csv}")

    print(f"  Tokenizing {len(df):,} prompts (this may take a few minutes)...")
    lengths = df["Prompt"].apply(lambda p: len(tokenizer(str(p))["input_ids"]))

    stats = {
        "n_rows": int(len(df)),
        "median": float(lengths.median()),
        "p95": float(lengths.quantile(0.95)),
        "p99": float(lengths.quantile(0.99)),
        "max": int(lengths.max()),
        "pct_over_512": float((lengths > 512).mean() * 100),
    }
    return stats


def measure_answer_lengths(train_csv: str, tokenizer) -> dict:
    """Token length distribution of output_text -- informs max_new_tokens."""
    df = pd.read_csv(train_csv)
    if "output_text" not in df.columns:
        raise ValueError(f"'output_text' column not found in {train_csv}")

    lengths = df["output_text"].apply(lambda t: len(tokenizer(str(t))["input_ids"]))
    stats = {
        "median": float(lengths.median()),
        "p95": float(lengths.quantile(0.95)),
        "p99": float(lengths.quantile(0.99)),
        "max": int(lengths.max()),
    }
    return stats


def measure_rule_stats(rules_csv: str) -> dict:
    """PCA confidence distribution and rule body length, from the mined
    rules CSV (Body, Head, Pca_Confidence columns) -- NOT the rendered
    rule_*.txt files, so this works before NL-instances scripts even run."""
    df = pd.read_csv(rules_csv)

    pca_col = next(
        (c for c in ["Pca_Confidence", "PCA_Confidence", "PCA Confidence", "pca_confidence"]
         if c in df.columns),
        None
    )
    if pca_col is None:
        raise ValueError(f"No PCA confidence column found in {rules_csv}")

    pca = pd.to_numeric(df[pca_col], errors="coerce").dropna()

    body_col = next((c for c in ["Body", "body"] if c in df.columns), None)
    if body_col:
        import re
        body_lengths = df[body_col].dropna().apply(
            lambda b: len(re.findall(r'\?\w+\s+\S+\s+\S+', str(b)))
        )
        max_body_len = int(body_lengths.max()) if len(body_lengths) else None
    else:
        max_body_len = None

    return {
        "n_rules": int(len(df)),
        "pca_median": float(pca.median()) if len(pca) else None,
        "pca_p25": float(pca.quantile(0.25)) if len(pca) else None,
        "pca_p75": float(pca.quantile(0.75)) if len(pca) else None,
        "max_body_atoms": max_body_len,
    }


def measure_train_pool_size(train_csv: str) -> int:
    df = pd.read_csv(train_csv, usecols=lambda c: c in ("Label",))
    return int(len(df))


def measure_walk_saturation(train_csv: str, configured_max_path_length: int) -> dict:
    """
    Baseline-only diagnostic. max_path_length controls DFS random-walk
    depth in _dfs_baseline_walk (path_length = randint(1, max_path_length-1))
    -- it has NO effect on CoT2/CoT3, whose structure comes from the mined
    rules, not from this parameter. Rule-body-atom-count is NOT a valid
    proxy for it (an earlier version of this script conflated the two).

    Instead: count ' has ' occurrences in input_text as a proxy for hops
    actually realized per walk, and check what fraction are sitting at the
    configured ceiling -- a high fraction suggests max_path_length is
    constraining walk diversity and could be raised.
    """
    df = pd.read_csv(train_csv, usecols=lambda c: c in ("input_text",))
    if "input_text" not in df.columns:
        return {"available": False}
    hop_counts = df["input_text"].str.count(r'\bhas\b')
    ceiling = max(configured_max_path_length - 1, 1)
    return {
        "available": True,
        "median_hops": float(hop_counts.median()),
        "p95_hops": float(hop_counts.quantile(0.95)),
        "configured_ceiling": ceiling,
        "pct_at_ceiling": float((hop_counts >= ceiling).mean() * 100),
    }


def recommend_max_path_length_from_saturation(sat_stats: dict, current: int) -> int:
    """Only raise max_path_length if a meaningful fraction of walks are
    actually hitting the current ceiling -- otherwise leave it alone rather
    than guessing."""
    if not sat_stats.get("available"):
        return current
    if sat_stats["pct_at_ceiling"] > 20.0:
        return current + 2
    return current


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def recommend_max_length(prompt_stats: dict) -> int:
    """Round the 99th percentile up to a clean multiple of 128, with a
    small safety margin, capped at a sane ceiling to avoid runaway cost."""
    raw = prompt_stats["p99"] * 1.05
    rec = round_up(raw, 128)
    return min(rec, 2048)


def recommend_max_new_tokens(answer_stats: dict) -> int:
    """Round the 99th percentile up to a clean multiple of 25, plus margin
    for the model's own token variance at inference vs. training text."""
    raw = answer_stats["p99"] * 1.2
    rec = round_up(raw, 25)
    return max(rec, 50)


def recommend_num_steps(train_pool_size: int, batch_size: int, grad_accum: int,
                        target_epochs: float = 3.0,
                        min_steps: int = 2000, max_steps: int = 30000) -> int:
    """Derive num_steps from actual data volume instead of a flat constant.
    effective_samples_per_step = batch_size * grad_accum."""
    eff_batch = batch_size * grad_accum
    steps_per_epoch = max(1, train_pool_size // eff_batch)
    rec = int(steps_per_epoch * target_epochs)
    return max(min_steps, min(rec, max_steps))


def recommend_pca_threshold(rule_stats: dict, default: float = 0.5) -> float:
    """
    Use the rules' own median PCA confidence as the threshold, so the
    POSITIVE/NEGATIVE split is roughly balanced for THIS dataset's rule
    quality distribution, rather than an absolute cutoff that may sit
    entirely above or below where this dataset's rules actually score.
    Falls back to the default if too few rules to be meaningful.

    NOTE: this value only takes effect if you re-run NL-instances-CoT2/3
    scripts with it -- prepare_data.py and utils.py only ever re-parse the
    POSITIVE/NEGATIVE label already baked into existing rule_*.txt files,
    they never recompute it from pca_threshold. Writing this into a config
    used only by prepare_data.py/step2/step3 has no effect on its own.
    """
    if rule_stats["pca_median"] is None or rule_stats["n_rules"] < 20:
        return default
    return round(rule_stats["pca_median"], 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NeSyKGLLM Config Auto-Tuner")
    parser.add_argument("--config", required=True, help="Path to existing config.json")
    parser.add_argument("--cot_version", default="CoT2", choices=["CoT2", "CoT3", "Baseline"],
                        help="Which train CSV to measure (default: CoT2). "
                             "'Baseline' measures train_data_without_rules.csv "
                             "for tuning step2_finetune_no_rules.py's config.")
    parser.add_argument("--output", default=None,
                        help="Where to write the tuned config (default: overwrite --config)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print measurements and recommendations without writing a file")
    parser.add_argument("--target_epochs", type=float, default=3.0,
                        help="Target training epochs for num_steps derivation (default: 3)")
    args = parser.parse_args()

    cfg = load_json(args.config)
    output_dir = cfg["output_dir"]
    dg = cfg.get("data_generation", {})
    tcfg = cfg.get("training", {})
    is_baseline = (args.cot_version == "Baseline")

    if is_baseline:
        train_csv = os.path.join(output_dir, "train_data_without_rules.csv")
    else:
        train_csv = os.path.join(output_dir, f"train_data_with_rules_{args.cot_version}.csv")

    if not os.path.exists(train_csv):
        print(f"ERROR: {train_csv} not found.")
        print("Run prepare_data.py first to generate training CSVs for this dataset.")
        sys.exit(1)

    print("=" * 70)
    print(f"CONFIG AUTO-TUNER — {cfg.get('data_dir', '?')} [{args.cot_version}]")
    print("=" * 70)

    hf_token = os.environ.get("HF_TOKEN") or cfg.get("huggingface_token")
    tokenizer, is_real = get_tokenizer(cfg["available_models"][cfg["model_key"]], hf_token)

    print("\n--- Measuring prompt (training input) token lengths ---")
    prompt_stats = measure_prompt_lengths(train_csv, tokenizer)
    for k, v in prompt_stats.items():
        print(f"  {k}: {v}")

    print("\n--- Measuring answer (output_text) token lengths ---")
    answer_stats = measure_answer_lengths(train_csv, tokenizer)
    for k, v in answer_stats.items():
        print(f"  {k}: {v}")

    train_pool_size = measure_train_pool_size(train_csv)
    print(f"\n--- Training pool size: {train_pool_size:,} rows ---")

    rec_max_length = recommend_max_length(prompt_stats)
    rec_max_new_tokens = recommend_max_new_tokens(answer_stats)
    rec_num_steps = recommend_num_steps(
        train_pool_size,
        batch_size=tcfg.get("per_device_train_batch_size", 1),
        grad_accum=tcfg.get("gradient_accumulation_steps", 8),
        target_epochs=args.target_epochs,
    )

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print(f"  training.max_length            : {tcfg.get('max_length', 512)}  ->  {rec_max_length}"
          f"{'  (tokenizer fallback — verify with real model)' if not is_real else ''}")
    print(f"  evaluation.max_new_tokens       : {cfg.get('evaluation', {}).get('max_new_tokens', 50)}  ->  {rec_max_new_tokens}"
          f"{'  (tokenizer fallback — verify with real model)' if not is_real else ''}")
    print(f"  training.num_steps              : {tcfg.get('num_steps', 20000)}  ->  {rec_num_steps}"
          f"  (target {args.target_epochs} epochs over {train_pool_size:,} rows)")

    new_cfg = copy.deepcopy(cfg)
    new_cfg.setdefault("training", {})["max_length"] = rec_max_length
    new_cfg.setdefault("training", {})["num_steps"] = rec_num_steps
    new_cfg.setdefault("evaluation", {})["max_new_tokens"] = rec_max_new_tokens

    if is_baseline:
        # max_path_length only controls the baseline DFS walk depth --
        # measure whether it's actually constraining walks, rather than
        # guessing from an unrelated statistic.
        configured_mpl = dg.get("max_path_length", 5)
        sat_stats = measure_walk_saturation(train_csv, configured_mpl)
        rec_max_path_length = recommend_max_path_length_from_saturation(sat_stats, configured_mpl)
        print(f"\n--- Baseline walk saturation ---")
        for k, v in sat_stats.items():
            print(f"  {k}: {v}")
        print(f"  data_generation.max_path_length : {configured_mpl}  ->  {rec_max_path_length}"
              f"{'  (>20% of walks at ceiling)' if rec_max_path_length != configured_mpl else '  (no evidence of constraint — unchanged)'}")
        new_cfg.setdefault("data_generation", {})["max_path_length"] = rec_max_path_length
    else:
        # pca_threshold is informational only here -- see docstring on
        # recommend_pca_threshold for why it doesn't take effect without
        # re-running the NL-instances rule-generation scripts.
        rules_csv = cfg.get("kg_sparql", {}).get("rules_csv")
        if rules_csv and os.path.exists(rules_csv):
            print(f"\n--- Measuring mined rule statistics ({rules_csv}) ---")
            rule_stats = measure_rule_stats(rules_csv)
            for k, v in rule_stats.items():
                print(f"  {k}: {v}")
            rec_pca_threshold = recommend_pca_threshold(rule_stats)
            print(f"  data_generation.pca_threshold   : {dg.get('pca_threshold', 0.5)}  ->  {rec_pca_threshold}"
                  f"  (median of {rule_stats['n_rules']} mined rules — "
                  f"REQUIRES re-running NL-instances scripts to take effect)")
            new_cfg.setdefault("data_generation", {})["pca_threshold"] = rec_pca_threshold
        else:
            print(f"\n--- Rules CSV not found ({rules_csv}) — skipping pca_threshold ---")
        print(f"\n  data_generation.max_path_length : not applicable to {args.cot_version} "
              f"(only affects the baseline DFS walk) — left unchanged.")

    print(f"\n  data_generation.relation_skew_threshold: NOT auto-set — "
          f"run prepare_data.py's filter_skewed_relations() diagnostic "
          f"at a few candidate thresholds and choose by inspection.")

    if args.dry_run:
        print("\n[dry run] No file written.")
        return

    new_cfg["huggingface_token"] = None   # never persist a real token

    out_path = args.output or args.config
    with open(out_path, "w") as f:
        json.dump(new_cfg, f, indent=4)
    print(f"\nTuned config written to: {out_path}")


if __name__ == "__main__":
    main()

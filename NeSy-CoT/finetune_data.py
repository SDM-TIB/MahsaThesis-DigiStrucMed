"""Data and token-budget checks for supervised rule fine-tuning."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def question_keys(frame, column="input_text"):
    if column not in frame or frame[column].isna().any():
        raise ValueError(f"Missing {column} values; cannot check split leakage")
    keys = frame[column].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    if keys.eq("").any():
        raise ValueError(f"Empty {column} values")
    return keys


def stratified_group_split(frame, fraction, seed=42, key_column="input_text"):
    """Reserve a fraction of each label's questions, keeping duplicates together."""
    if not 0 < fraction < 1:
        raise ValueError("Validation/test fraction must be between 0 and 1")
    if "Label" not in frame or not frame["Label"].isin([0, 1]).all():
        raise ValueError("All rows must have binary ground-truth Label values")
    keys = question_keys(frame, key_column)
    groups = pd.DataFrame({"key": keys.to_numpy(), "Label": frame["Label"].to_numpy()})
    if groups.groupby("key")["Label"].nunique().gt(1).any():
        raise ValueError("The same question has contradictory ground-truth labels")
    unique = groups.drop_duplicates("key")
    rng = np.random.default_rng(seed)
    held_keys = set()
    for label in (0, 1):
        pool = unique.loc[unique["Label"].eq(label), "key"].to_numpy().copy()
        if len(pool) < 2:
            raise ValueError(f"Need at least two distinct questions for Label={label}")
        rng.shuffle(pool)
        count = min(len(pool) - 1, max(1, round(len(pool) * fraction)))
        held_keys.update(pool[:count])
    held = keys.isin(held_keys).to_numpy()
    return (frame.iloc[np.flatnonzero(~held)].reset_index(drop=True),
            frame.iloc[np.flatnonzero(held)].reset_index(drop=True))


def audit_frame(frame, name, require_targets=False):
    if frame.empty or "Label" not in frame or not frame["Label"].isin([0, 1]).all():
        raise ValueError(f"{name}: empty dataset or invalid/missing Label values")
    if set(frame["Label"].unique()) != {0, 1}:
        raise ValueError(f"{name}: both yes (1) and no (0) are required")
    question_keys(frame)
    if "Prompt" not in frame or frame["Prompt"].isna().any():
        raise ValueError(f"{name}: missing prompts")
    stats = {"rows": len(frame), "yes": int(frame["Label"].eq(1).sum()),
             "no": int(frame["Label"].eq(0).sum())}
    if require_targets:
        if "output_text" not in frame or frame["output_text"].isna().any():
            raise ValueError(f"{name}: missing output_text")
        answers = frame["output_text"].str.extract(
            r"(?i)\bthe answer is (yes|no)\.\s*$", expand=False)
        if answers.isna().any() or not answers.eq(frame["Label"].map({0: "no", 1: "yes"})).all():
            raise ValueError(f"{name}: terminal answer is missing or disagrees with Label")
        for prompt, target in zip(frame["Prompt"], frame["output_text"]):
            if "###Response:" not in prompt or prompt.rsplit("###Response:", 1)[1].strip() != target.strip():
                raise ValueError(f"{name}: Prompt response differs from output_text")
        classification = frame["output_text"].str.extract(
            r"classified as (POSITIVE|NEGATIVE)", expand=False)
        stats["rule_classification"] = {str(k): int(v) for k, v in classification.value_counts().items()}
    print(f"  {name}: {stats['rows']:,} rows, yes={stats['yes']:,}, no={stats['no']:,}")
    return stats


def assert_disjoint(**frames):
    names = list(frames)
    keys = {name: set(question_keys(frame)) for name, frame in frames.items()}
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            count = len(keys[name] & keys[other])
            if count:
                raise ValueError(f"Question leakage between {name} and {other}: {count} groups")


def tokenize_responses(batch, tokenizer, max_length):
    """Keep complete examples and compute loss only on their answer/reasoning."""
    if not tokenizer.is_fast:
        raise ValueError("Response-only training requires a fast tokenizer with offsets")
    encoded = tokenizer(batch["Prompt"], truncation=False, padding=False,
                        return_offsets_mapping=True)
    labels = []
    for i, (prompt, offsets) in enumerate(zip(batch["Prompt"], encoded.pop("offset_mapping"))):
        if "###Response:" not in prompt:
            raise ValueError("Training prompt has no ###Response: delimiter")
        boundary = prompt.rindex("###Response:") + len("###Response:")
        ids = encoded["input_ids"][i]
        targets = [token if end > boundary else -100 for token, (_, end) in zip(ids, offsets)]
        if all(token == -100 for token in targets):
            raise ValueError("Training example contains no answer tokens")
        if not ids or ids[-1] != tokenizer.eos_token_id:
            ids.append(tokenizer.eos_token_id)
            encoded["attention_mask"][i].append(1)
            targets.append(tokenizer.eos_token_id)
        else:
            targets[-1] = tokenizer.eos_token_id
        if len(ids) > max_length:
            raise ValueError(f"Example needs {len(ids)} tokens; training.max_length={max_length}. "
                             "Increase the budget; no examples were silently truncated.")
        labels.append(targets)
    encoded["labels"] = labels
    return encoded


def _lengths(texts, tokenizer):
    lengths = []
    texts = list(texts)
    for start in range(0, len(texts), 128):
        encoded = tokenizer(texts[start:start + 128], truncation=False, padding=False)
        lengths.extend(map(len, encoded["input_ids"]))
    return np.asarray(lengths)


def audit_token_budgets(tokenizer, cfg, train, validation, matched, clean):
    budgets = cfg["training"], cfg["evaluation"]
    report = {}
    for name, frame in (("train", train), ("validation", validation), ("matched", matched), ("clean", clean)):
        training = name in ("train", "validation")
        texts = frame["Prompt"] if training else frame["Prompt"].map(
            lambda prompt: prompt.split("###Response:", 1)[0] + "###Response:"
            if "###Response:" in prompt else prompt)
        sizes = _lengths(texts, tokenizer) + (1 if training else 0)
        limit = budgets[0]["max_length"] if training else budgets[1]["max_input_tokens"]
        over = int((sizes > limit).sum())
        report[name] = {"max_tokens": int(sizes.max()), "p95_tokens": int(np.percentile(sizes, 95)),
                        "budget": limit, "over_budget": over}
        if over:
            raise ValueError(f"{name}: {over} examples exceed {limit} tokens; maximum={sizes.max()}. "
                             "Increase the token budget before training.")
    target_sizes = _lengths(pd.concat([train["output_text"], validation["output_text"]]), tokenizer) + 1
    report["answers"] = {"max_tokens": int(target_sizes.max()), "budget": budgets[1]["max_new_tokens"]}
    if target_sizes.max() > budgets[1]["max_new_tokens"]:
        raise ValueError(f"evaluation.max_new_tokens={budgets[1]['max_new_tokens']} is shorter "
                         f"than the longest reference answer ({target_sizes.max()} tokens)")
    return report


def write_audit(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

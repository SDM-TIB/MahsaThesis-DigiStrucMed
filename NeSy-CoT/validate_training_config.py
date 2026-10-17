#!/usr/bin/env python3
"""Audit all CoT2 rows and tokenize them without loading model weights."""
import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from dataset_io import audit_dataset_csv
from finetune_data import (audit_frame, assert_disjoint, stratified_group_split,
                           audit_token_budgets, write_audit)
from utils import parse_rule_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", action="store_true", help="Also audit the optional baseline dataset")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    paths = cfg["pregenerated_data"]
    frames = {}
    for name, key in (("source_train", "train_with_rules_cot2_csv"),
                      ("matched_test", "test_shared_cot2_csv"),
                      ("clean_test", "test_eval_clean_csv")):
        audit_dataset_csv(paths[key])
        frames[name] = pd.read_csv(paths[key])
    source, matched, clean = (frames[name] for name in ("source_train", "matched_test", "clean_test"))
    report = {name: audit_frame(frame, name, require_targets=name != "clean_test")
              for name, frame in frames.items()}
    assert_disjoint(training=source, test=matched)
    assert_disjoint(training=source, clean_test=clean)
    if list(matched["Label"]) != list(clean["Label"]) or list(matched["input_text"]) != list(clean["input_text"]):
        raise ValueError("Matched and clean test rows are not aligned")
    train, validation = stratified_group_split(
        source, cfg["training"]["validation_fraction"], cfg.get("seed", 42))
    assert_disjoint(training=train, validation=validation, test=matched)
    report["train"] = audit_frame(train, "train", require_targets=True)
    report["validation"] = audit_frame(validation, "validation", require_targets=True)
    report["question_overlap"] = 0
    report["model"] = cfg["available_models"][cfg["model_key"]]
    rules = list(Path(cfg["rules_dir_cot2"]).glob("rule_*.txt"))
    if not rules:
        raise ValueError("No source rules available for the row-preservation audit")
    source_rows = sum(len(parse_rule_file(str(path))["instances"]) for path in rules)
    report["source_rule_instances"] = source_rows
    report["preserved_rows"] = len(source) + len(matched)
    if report["preserved_rows"] != source_rows:
        raise ValueError(f"Expected {source_rows} instances, found {report['preserved_rows']} train/test rows")
    tokenizer = AutoTokenizer.from_pretrained(report["model"], use_fast=True)
    report["token_budgets"] = audit_token_budgets(tokenizer, cfg, train, validation, matched, clean)
    if args.baseline:
        baseline_path = paths["train_without_rules_csv"]
        audit_dataset_csv(baseline_path)
        baseline = pd.read_csv(baseline_path)
        report["baseline_source"] = audit_frame(baseline, "baseline", require_targets=True)
        baseline_train, baseline_val = stratified_group_split(
            baseline, cfg["training"]["validation_fraction"], cfg.get("seed", 42))
        assert_disjoint(training=baseline_train, validation=baseline_val, test=clean)
        report["baseline_token_budgets"] = audit_token_budgets(
            tokenizer, cfg, baseline_train, baseline_val, clean, clean)
    effective_batch = cfg["training"]["per_device_train_batch_size"] * cfg["training"]["gradient_accumulation_steps"]
    report["approximate_optimizer_steps"] = (
        (len(train) + effective_batch - 1) // effective_batch * cfg["training"]["num_train_epochs"])
    output = Path(cfg["output_dir"]) / "preflight_CoT2.json"
    write_audit(output, report)
    print(json.dumps(report, indent=2))
    print(f"PASS: all source instances accounted for, no question overlap or truncation; audit: {output}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()

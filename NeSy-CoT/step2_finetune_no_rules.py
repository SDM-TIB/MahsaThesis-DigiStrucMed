#!/usr/bin/env python3
"""
NeSyKGLLM - Step 2: Fine-tune WITHOUT Rules
==============================================
Fine-tunes the model on training data generated WITHOUT symbolic rule context,
then evaluates on the test set.

Usage:
    python step2_finetune_no_rules.py --config config.json
"""

import argparse
import os
from functools import partial

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from dataset_io import audit_dataset_csv
from finetune_data import (audit_frame, assert_disjoint, stratified_group_split,
                           tokenize_responses, audit_token_budgets, write_audit)

from utils import (
    load_config,
    set_all_seeds,
    login_huggingface,
    create_bnb_config,
    load_model,
    preprocess_batch,
    fine_tune_model,
    evaluate_model,
    print_results,
    save_results_json,
)


def main():
    parser = argparse.ArgumentParser(
        description="NeSyKGLLM - Fine-tune WITHOUT Rules"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg["seed"])
    login_huggingface(cfg["huggingface_token"])

    output_dir = cfg["output_dir"]
    eval_cfg = cfg.get("evaluation", {})
    tcfg = cfg.get("training", {})
    model_tag = cfg["model_key"].replace("-", "_")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    pregen = cfg.get("pregenerated_data", {})
    if pregen.get("use_pregenerated") and pregen.get("train_without_rules_csv"):
        train_path = pregen["train_without_rules_csv"]
    else:
        train_path = os.path.join(output_dir, "train_data_without_rules.csv")

    # test_eval_clean.csv is the canonical shared test: path facts + question only,
    # no rule context, no symbolic tags — identical for Baseline, CoT2, CoT3.
    test_path = (
        pregen.get("test_eval_clean_csv")
        or os.path.join(output_dir, "test_eval_clean.csv")
    )
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Shared eval file not found: {test_path}\n"
            "Run prepare_data.py first to generate test_eval_clean.csv."
        )

    print(f"Loading training data: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading test data:     {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    validation_df = None
    audit = {}
    if tcfg.get("strict_data_checks", False):
        audit_dataset_csv(train_path)
        audit_dataset_csv(test_path)
        audit["source_train"] = audit_frame(train_df, "Baseline source", require_targets=True)
        audit["test"] = audit_frame(test_df, "Baseline test")
        assert_disjoint(training=train_df, test=test_df)
    if tcfg.get("validation_fraction", 0):
        train_df, validation_df = stratified_group_split(
            train_df, tcfg["validation_fraction"], seed=cfg.get("seed", 42))
        assert_disjoint(training=train_df, validation=validation_df, test=test_df)
        audit["train"] = audit_frame(train_df, "Baseline training", require_targets=True)
        audit["validation"] = audit_frame(validation_df, "Baseline validation", require_targets=True)
    if tcfg.get("fail_on_truncation", False):
        if validation_df is None:
            raise ValueError("Strict token audit requires a validation split")
        audit_tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
        audit["token_budgets"] = audit_token_budgets(
            audit_tokenizer, cfg, train_df, validation_df, test_df, test_df)
        write_audit(os.path.join(output_dir, "data_audit_Baseline.json"), audit)
        del audit_tokenizer

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: FINE-TUNING WITHOUT RULES")
    print("=" * 70)
    print(f"Model: {cfg['model_name']}")

    bnb_config = create_bnb_config(cfg)
    model, tokenizer = load_model(
        cfg["model_name"], bnb_config, cfg.get("gpu_max_memory_mb", 40960)
    )

    # ------------------------------------------------------------------
    # Prepare dataset
    # ------------------------------------------------------------------
    max_length = tcfg.get("max_length", 512)
    def make_dataset(frame):
        dataset = Dataset.from_pandas(frame, preserve_index=False)
        return dataset.map(
            lambda batch: (tokenize_responses(batch, tokenizer, max_length)
                           if tcfg.get("response_only_loss", False)
                           else preprocess_batch(batch, tokenizer, max_length=max_length)),
            batched=True, remove_columns=dataset.column_names)
    train_dataset = make_dataset(train_df)
    validation_dataset = make_dataset(validation_df) if validation_df is not None else None

    # ------------------------------------------------------------------
    # Fine-tune
    # ------------------------------------------------------------------
    model_output_dir = os.path.join(output_dir, f"finetuned_{model_tag}_baseline")
    model = fine_tune_model(model, tokenizer, train_dataset, model_output_dir, cfg,
                            validation_dataset=validation_dataset)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print("\nEVALUATING FINE-TUNED MODEL (WITHOUT RULES)")
    results = evaluate_model(
        model,
        tokenizer,
        test_df,
        max_samples=eval_cfg.get("max_samples", 2000),
        max_new_tokens=eval_cfg.get("max_new_tokens", 150),
        max_input_tokens=eval_cfg.get("max_input_tokens", 2048),
        sampling=eval_cfg.get("sampling", "head"),
        seed=cfg.get("seed", 42),
        require_explicit_answer=eval_cfg.get("require_explicit_answer", False),
        fail_on_truncation=eval_cfg.get("fail_on_truncation", False),
    )
    print_results("FINE-TUNED MODEL (WITHOUT RULES) RESULTS:", results)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path = os.path.join(output_dir, "finetune_no_rules_results.json")
    save_results_json(results_path, {
        "model": cfg["model_key"],
        "model_name": cfg["model_name"],
        "step": "finetune_without_rules",
        "model_saved_to": model_output_dir,
        "data_audit": audit,
        "metrics": results,
    })

    # Cleanup
    del model
    torch.cuda.empty_cache()
    print("\nFine-tuning (without rules) complete!")


if __name__ == "__main__":
    main()

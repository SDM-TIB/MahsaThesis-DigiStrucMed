#!/usr/bin/env python3
"""
NeSyKGLLM - Step 1: Evaluate Base Model
=========================================
Loads the pre-trained model (no fine-tuning) and evaluates it on the test set.

Usage:
    python step1_evaluate_base.py --config config.json
"""

import argparse
import os

import pandas as pd
import torch

from utils import (
    load_config,
    set_all_seeds,
    login_huggingface,
    create_bnb_config,
    load_model,
    evaluate_model,
    print_results,
    save_results_json,
)


def main():
    parser = argparse.ArgumentParser(description="NeSyKGLLM - Evaluate Base Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg["seed"])
    login_huggingface(cfg["huggingface_token"])

    output_dir = cfg["output_dir"]
    eval_cfg = cfg.get("evaluation", {})

    # ------------------------------------------------------------------
    # Load test data  (shared cross-format eval — same file used by all steps)
    # ------------------------------------------------------------------
    pregen = cfg.get("pregenerated_data", {})
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

    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Test samples: {len(test_df)}")

    # ------------------------------------------------------------------
    # Load and evaluate base model
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: EVALUATING BASE MODEL")
    print("=" * 70)
    print(f"Model: {cfg['model_name']}")

    bnb_config = create_bnb_config(cfg)
    model, tokenizer = load_model(
        cfg["model_name"], bnb_config, cfg.get("gpu_max_memory_mb", 40960)
    )

    results = evaluate_model(
        model,
        tokenizer,
        test_df,
        max_samples=eval_cfg.get("max_samples", 2000),
        max_new_tokens=eval_cfg.get("max_new_tokens", 150),
        max_input_tokens=eval_cfg.get("max_input_tokens", 2048),
    )

    print_results("BASE MODEL RESULTS:", results)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_path = os.path.join(output_dir, "base_model_results.json")
    save_results_json(results_path, {
        "model": cfg["model_key"],
        "model_name": cfg["model_name"],
        "step": "base_model_evaluation",
        "metrics": results,
    })

    # Cleanup
    del model
    torch.cuda.empty_cache()
    print("\nBase model evaluation complete!")


if __name__ == "__main__":
    main()

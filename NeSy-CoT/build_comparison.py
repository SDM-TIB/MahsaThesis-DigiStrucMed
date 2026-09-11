#!/usr/bin/env python3
"""
NeSyKGLLM - Build CoT2 vs CoT3 Comparison
=============================================
Reconstructs comparison_CoT2_vs_CoT3.json from separately-run
finetune_with_rules_CoT2_results.json / finetune_with_rules_CoT3_results.json
files.

WHY THIS EXISTS
---------------
step3_finetune_with_rules.py only builds comparison_CoT2_vs_CoT3.json when
BOTH versions are trained in the same process (--cot_version both), because
it needs both results dicts in memory at once. But --cot_version both loads
a single shared config for both versions -- no per-version max_length /
num_steps -- which reintroduces exactly the kind of cross-version value
mismatch this pipeline was just fixed for at the dataset level (see
auto_configure.py). Since CoT2 and CoT3 prompts have measurably different
token-length profiles (CoT3 carries extra [VALID]/[INVALID] + SHACL
reasoning text), they should be tuned and run separately:

    python step3_finetune_with_rules.py --config config_tuned_CoT2.json --cot_version CoT2
    python step3_finetune_with_rules.py --config config_tuned_CoT3.json --cot_version CoT3
    python build_comparison.py --output_dir ./outputs/FB15K-237/

This script reads both resulting JSON files independently and writes the
same comparison_CoT2_vs_CoT3.json step3 would have produced.

Usage:
    python build_comparison.py --output_dir ./outputs/FB15K-237/
"""

import argparse
import json
import os


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild comparison_CoT2_vs_CoT3.json from separate CoT2/CoT3 runs"
    )
    parser.add_argument("--output_dir", required=True,
                        help="Dataset output_dir containing both results JSON files")
    args = parser.parse_args()

    cot2_path = os.path.join(args.output_dir, "finetune_with_rules_CoT2_results.json")
    cot3_path = os.path.join(args.output_dir, "finetune_with_rules_CoT3_results.json")

    missing = [p for p in (cot2_path, cot3_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing results file(s):\n  " + "\n  ".join(missing) +
            "\nRun step3_finetune_with_rules.py for both CoT2 and CoT3 first "
            "(each under its own tuned config)."
        )

    cot2 = load_results(cot2_path)
    cot3 = load_results(cot3_path)

    # Primary reported metric in each results file is metrics_clean_eval
    # (fair cross-format comparison -- see step3's own "metrics" field).
    cot2_metrics = cot2.get("metrics_clean_eval", cot2.get("metrics", {}))
    cot3_metrics = cot3.get("metrics_clean_eval", cot3.get("metrics", {}))

    print("=" * 70)
    print("COMPARISON: CoT2 vs CoT3  (clean cross-format eval — no tags)")
    print("=" * 70)
    print(f"{'Metric':<15} {'CoT2 (v1)':>12} {'CoT3 (v2)':>12} {'Delta':>10}")
    print("-" * 50)
    for metric in ["accuracy", "f1_score", "precision", "recall"]:
        v2 = cot2_metrics.get(metric, 0)
        v3 = cot3_metrics.get(metric, 0)
        delta = v3 - v2
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<15} {v2:>12.4f} {v3:>12.4f} {sign}{delta:>9.4f}")

    if cot2.get("model_name") != cot3.get("model_name"):
        print(f"\n  WARNING: model_name differs between runs "
              f"({cot2.get('model_name')} vs {cot3.get('model_name')}) — "
              f"comparison may not be apples-to-apples.")

    comp_path = os.path.join(args.output_dir, "comparison_CoT2_vs_CoT3.json")
    with open(comp_path, "w") as f:
        json.dump({
            "evaluation": "clean_cross_format",
            "eval_file": "test_eval_clean.csv",
            "note": (
                "All models evaluated on path facts + question only — "
                "no rule context, no symbolic tags visible to the model. "
                "CoT2 and CoT3 were fine-tuned in SEPARATE runs, each under "
                "its own auto-tuned config (max_length/num_steps measured "
                "independently per version) — see cot2_source_results / "
                "cot3_source_results for provenance."
            ),
            "cot2_source_results": cot2_path,
            "cot3_source_results": cot3_path,
            "CoT2_metrics": {
                k: v for k, v in cot2_metrics.items() if k not in ("y_true", "y_pred")
            },
            "CoT3_metrics": {
                k: v for k, v in cot3_metrics.items() if k not in ("y_true", "y_pred")
            },
        }, f, indent=2)
    print(f"\nComparison written to: {comp_path}")


if __name__ == "__main__":
    main()

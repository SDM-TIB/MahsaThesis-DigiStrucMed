"""
evaluate_openai.py

Evaluate OpenAI-compatible models (OpenAI or NVIDIA NIM) on the NeSyKGLLM test set.
Computes the same metrics used for fine-tuned LLaMA models:
  Accuracy, Precision, Recall, F1 (binary yes/no + weighted)

Input CSV columns expected:
  - input_text   : clean CoT path question — sent to the model
  - output_text  : ground truth, ending with 'The answer is yes/no.' — NOT sent to model
  - Prompt       : full fine-tuning prompt — ignored (contains answer, would leak GT)

Usage (NVIDIA NIM):
    python evaluate_openai.py \
        --test_file outputs/SynthLC-1000/test_data_shared.csv \
        --model meta/llama-3.3-70b-instruct \
        --api_key nvapi-xxxx... \
        --provider nvidia \
        --variant baseline \
        --output_dir results/nvidia \
        --max_samples 500

Usage (OpenAI):
    python evaluate_openai.py \
        --test_file outputs/SynthLC-1000/test_data_shared.csv \
        --model gpt-4.1 \
        --api_key sk-xxxx... \
        --provider openai \
        --variant baseline \
        --output_dir results/openai \
        --max_samples 500
"""

import os
import re
import json
import time
import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider base URLs
# ---------------------------------------------------------------------------

PROVIDER_URLS = {
    "openai":  "https://api.openai.com/v1",
    "nvidia":  "https://integrate.api.nvidia.com/v1",
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a knowledge graph reasoning assistant. "
    "Given a question about entities and their relationships, determine whether "
    "the stated relationship is supported by the evidence. "
    "Respond with exactly one word: 'yes' or 'no'."
)

# ---------------------------------------------------------------------------
# Prompt construction — sends ONLY input_text (CoT path), no symbolic tags
# ---------------------------------------------------------------------------

def build_prompt(sample: dict) -> str:
    """
    Send only the 'input_text' column to the model.
    This is the clean CoT path question with no POSITIVE/NEGATIVE or VALID/INVALID tags.
    Ground truth is NOT included.
    """
    return sample.get("input_text", "").strip()

# ---------------------------------------------------------------------------
# Ground truth extraction from output_text
# ---------------------------------------------------------------------------

def extract_ground_truth(output_text: str) -> str | None:
    """
    Extract yes/no from output_text which ends with 'The answer is yes/no.'
    """
    text = (output_text or "").strip().lower()
    match = re.search(r"the answer is (yes|no)", text)
    if match:
        return match.group(1)
    # Fallback: plain yes/no at end
    if text.endswith("yes") or text.endswith("yes."):
        return "yes"
    if text.endswith("no") or text.endswith("no."):
        return "no"
    return None

# ---------------------------------------------------------------------------
# Answer extraction from model response
# ---------------------------------------------------------------------------

def extract_answer(response_text: str) -> str:
    """
    Normalise model output to 'yes' or 'no'.
    Handles verbose responses by scanning for the first clear signal.
    """
    text = (response_text or "").strip().lower()
    # Check first word
    first_word = text.split()[0].strip(".,!?") if text.split() else ""
    if first_word in ("yes", "true", "1"):
        return "yes"
    if first_word in ("no", "false", "0"):
        return "no"
    # Scan full response
    if re.search(r"\byes\b", text):
        return "yes"
    if re.search(r"\bno\b", text):
        return "no"
    logger.warning(f"Could not extract yes/no from: '{response_text[:80]}'")
    return "unknown"

# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

def call_api(client: OpenAI, model: str, prompt: str,
             max_new_tokens: int, retry_delay: float) -> str:
    """
    Call the model API with exponential backoff on failure.
    Returns raw response string, or empty string on total failure.
    """
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=max_new_tokens,
                temperature=0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            wait = retry_delay * (2 ** attempt)
            logger.warning(f"API error (attempt {attempt+1}/5): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    logger.error("All retry attempts failed for this sample.")
    return ""

# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: list, y_pred: list) -> dict:
    """
    Compute Accuracy, Precision, Recall, F1 (binary + weighted).
    Mirrors the metrics used in the LLaMA fine-tuned model evaluation.
    """
    labels = ["yes", "no"]

    acc = accuracy_score(y_true, y_pred)

    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="binary", pos_label="yes", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy":           round(float(acc),   4),
        "precision_yes":      round(float(p_bin),  4),
        "recall_yes":         round(float(r_bin),  4),
        "f1_yes":             round(float(f1_bin), 4),
        "precision_weighted": round(float(p_w),    4),
        "recall_weighted":    round(float(r_w),    4),
        "f1_weighted":        round(float(f1_w),   4),
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "n_total":   len(y_true),
        "n_unknown": sum(1 for p in y_pred if p == "unknown"),
    }

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args):
    # --- Build client ---
    base_url = PROVIDER_URLS.get(args.provider)
    if base_url is None:
        raise ValueError(f"Unknown provider '{args.provider}'. Choose 'openai' or 'nvidia'.")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("No API key provided. Use --api_key or set OPENAI_API_KEY / NVIDIA_API_KEY.")

    client = OpenAI(base_url=base_url, api_key=api_key)
    logger.info(f"Provider : {args.provider}  ({base_url})")
    logger.info(f"Model    : {args.model}")

    # --- Load test file ---
    test_path = Path(args.test_file)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    if test_path.suffix == ".csv":
        df = pd.read_csv(test_path)
        samples = df.to_dict(orient="records")
    elif test_path.suffix == ".jsonl":
        with open(test_path) as f:
            samples = [json.loads(line) for line in f if line.strip()]
    elif test_path.suffix == ".json":
        with open(test_path) as f:
            samples = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {test_path.suffix}")

    if args.max_samples:
        samples = samples[:args.max_samples]
    logger.info(f"Loaded {len(samples)} samples from {test_path.name}")

    # --- Output directory ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "-")
    pred_path    = out_dir / f"{safe_model}_{args.variant}_predictions.csv"
    metrics_path = out_dir / f"{safe_model}_{args.variant}_metrics.json"

    # --- Evaluation loop ---
    results = []
    skipped = 0

    for i, sample in enumerate(samples):
        prompt = build_prompt(sample)
        gt     = extract_ground_truth(str(sample.get("output_text", "")))

        if not prompt:
            logger.warning(f"Sample {i}: empty input_text, skipping.")
            skipped += 1
            continue
        if gt is None:
            logger.warning(f"Sample {i}: could not extract ground truth, skipping.")
            skipped += 1
            continue

        raw_response = call_api(
            client, args.model, prompt,
            max_new_tokens=args.max_new_tokens,
            retry_delay=args.retry_delay,
        )
        pred = extract_answer(raw_response)

        results.append({
            "sample_id":    i,
            "input_text":   prompt[:200],   # truncated for readability
            "ground_truth": gt,
            "prediction":   pred,
            "correct":      int(pred == gt),
            "raw_response": raw_response[:200],
        })

        if (i + 1) % 50 == 0:
            so_far = [r for r in results if r["prediction"] != "unknown"]
            if so_far:
                acc = sum(r["correct"] for r in so_far) / len(so_far)
                logger.info(f"Progress: {i+1}/{len(samples)} | Running accuracy: {acc:.3f}")

        time.sleep(args.request_delay)

    if not results:
        logger.error("No valid results collected. Check your test file format.")
        return

    # --- Save predictions ---
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved → {pred_path}")

    # --- Compute & save metrics ---
    valid = pred_df[pred_df["prediction"] != "unknown"]
    if len(valid) == 0:
        logger.error("No valid predictions to compute metrics from.")
        return

    metrics = compute_metrics(
        valid["ground_truth"].tolist(),
        valid["prediction"].tolist(),
    )
    metrics["model"]   = args.model
    metrics["variant"] = args.variant
    metrics["provider"] = args.provider
    metrics["skipped"] = skipped

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved → {metrics_path}")
    logger.info("=" * 50)
    logger.info(f"  Accuracy          : {metrics['accuracy']}")
    logger.info(f"  F1 (yes, binary)  : {metrics['f1_yes']}")
    logger.info(f"  F1 (weighted)     : {metrics['f1_weighted']}")
    logger.info(f"  Precision (yes)   : {metrics['precision_yes']}")
    logger.info(f"  Recall (yes)      : {metrics['recall_yes']}")
    logger.info(f"  Unknown preds     : {metrics['n_unknown']} / {metrics['n_total']}")
    logger.info("=" * 50)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate OpenAI-compatible models on NeSyKGLLM test set.")
    p.add_argument("--test_file",      required=True,  help="Path to test CSV/JSON/JSONL file")
    p.add_argument("--model",          required=True,  help="Model name (e.g. meta/llama-3.3-70b-instruct)")
    p.add_argument("--api_key",        default=None,   help="API key (or set OPENAI_API_KEY / NVIDIA_API_KEY env var)")
    p.add_argument("--provider",       default="nvidia", choices=["openai", "nvidia"],
                                                        help="API provider (default: nvidia)")
    p.add_argument("--variant",        default="baseline", help="Label for output filenames (baseline/cot2/cot3)")
    p.add_argument("--output_dir",     default="results/nvidia", help="Directory to save predictions and metrics")
    p.add_argument("--max_samples",    type=int, default=None, help="Cap number of samples (cost control)")
    p.add_argument("--max_new_tokens", type=int, default=50,   help="Max tokens in model response")
    p.add_argument("--request_delay",  type=float, default=0.1, help="Seconds between API calls")
    p.add_argument("--retry_delay",    type=float, default=2.0, help="Base retry delay in seconds")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)

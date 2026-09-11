"""Generate BRINK predictions with the rules-fine-tuned model, BRINK-style.

Following the BRINK protocol, the question is passed to the model as-is --
no instruction wrapper and no rule context are added to the prompt, since the
adapter was fine-tuned on rule-augmented data and has already internalized the
rules. The model's raw generated text is saved unmodified as ``raw_output``;
splitting into candidate answers and normalization happen inside
``evaluate_brink.py``, not here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_ADAPTER = Path("model/qKG/finetuned_LLaMA_3.2_1B_with_rules_CoT2")
DEFAULT_GOLD = Path("data/qKG/question/val_cleaned_final_gold.json")
DEFAULT_OUTPUT = Path("data/qKG/question/val_cleaned_final_brink_pred.json")


def save_predictions(path: Path, predictions: list[dict[str, str]]) -> None:
    """Atomically checkpoint predictions using the evaluation JSON schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def load_existing_predictions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        predictions = json.load(file)
    for item in predictions:
        if set(item) != {"id", "raw_output"}:
            raise ValueError(f"Invalid prediction object in {path}: {item!r}")
        item["id"] = str(item["id"])
        item["raw_output"] = str(item["raw_output"])
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate BRINK predictions from the rules-fine-tuned model using "
            "the raw question only, with no prompt engineering."
        )
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help=(
            "Model precision on CPU. float16 uses about half the memory of "
            "float32 (default: float16)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing predictions and generate only missing IDs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.gold.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    predictions = load_existing_predictions(args.output) if args.resume else []
    completed_ids = {item["id"] for item in predictions}
    pending = [item for item in questions if str(item["id"]) not in completed_ids]
    if not pending:
        print(f"Nothing to generate; all {len(predictions)} IDs are complete.")
        return

    model_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    print(
        f"Loading model once for {len(pending)} pending question(s) "
        f"using {args.dtype}..."
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=model_dtype,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    for index, item in enumerate(pending, start=1):
        question_id = str(item["id"])
        messages = [{"role": "user", "content": item["question"]}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = output[0, inputs["input_ids"].shape[1] :]
        raw_output = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        predictions.append({"id": question_id, "raw_output": raw_output})
        save_predictions(args.output, predictions)
        print(
            f"[{index}/{len(pending)}] {question_id}: {raw_output}", flush=True
        )

    print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()

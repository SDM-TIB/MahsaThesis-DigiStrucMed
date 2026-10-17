"""Run manually from NeSy-CoT: python tests/check_training_runtime.py.

Checks the real Trainer/PEFT/collator API on a cached small model. It does not
load Qwen3B or write into the real experiment's output directory.
"""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import Dataset
import pandas as pd
from finetune_data import stratified_group_split, tokenize_responses, audit_frame
from utils import create_bnb_config, load_model, fine_tune_model, set_all_seeds


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "T4B-config.json").read_text(encoding="utf-8"))
    source = pd.read_csv(root / cfg["pregenerated_data"]["train_with_rules_cot2_csv"])
    train, validation = stratified_group_split(source, 0.1)
    train = train.groupby("Label", group_keys=False).head(4)
    validation = validation.groupby("Label", group_keys=False).head(2)
    audit_frame(train, "Runtime training", require_targets=True)
    audit_frame(validation, "Runtime validation", require_targets=True)
    cfg["training"].update(num_train_epochs=2, num_steps=-1, max_length=1024,
                           gradient_accumulation_steps=1, logging_steps=1,
                           resume_from_checkpoint=False, warmup_ratio=0,
                           early_stopping_patience=1)
    cfg["lora"].update(r=4, lora_alpha=8)
    set_all_seeds(42)
    model, tokenizer = load_model("HuggingFaceTB/SmolLM2-135M-Instruct", create_bnb_config(cfg), 1400)
    def make_dataset(frame):
        dataset = Dataset.from_pandas(frame, preserve_index=False)
        return dataset.map(lambda batch: tokenize_responses(batch, tokenizer, 1024),
                           batched=True, remove_columns=dataset.column_names)
    with tempfile.TemporaryDirectory(prefix="nesy-validation-runtime-") as temporary:
        fine_tune_model(model, tokenizer, make_dataset(train), temporary, cfg,
                        validation_dataset=make_dataset(validation))
        states = list(Path(temporary).glob("checkpoint-*/trainer_state.json"))
        assert states, "No epoch checkpoint was written"
        state = json.loads(states[-1].read_text(encoding="utf-8"))
        assert state["best_model_checkpoint"], "Best checkpoint was not selected"
        assert any("eval_loss" in row for row in state["log_history"]), "Validation was not run"
        assert (Path(temporary) / "adapter_model.safetensors").is_file()
        assert (Path(temporary) / "tokenizer_config.json").is_file()
        print("PASS: actual response-only training, epoch validation, best checkpoint, adapter and tokenizer save")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()

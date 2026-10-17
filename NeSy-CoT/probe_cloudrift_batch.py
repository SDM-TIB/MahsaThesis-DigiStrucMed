"""Select a memory-safe microbatch on the VM while keeping effective batch 8."""
import gc
import json
from pathlib import Path
import time

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForSeq2Seq
from finetune_data import tokenize_responses
from utils import create_bnb_config, find_all_linear_names, load_config, load_model


def main():
    cfg = load_config("T4B-config.json")
    model, tokenizer = load_model(cfg["model_name"], create_bnb_config(cfg), cfg["gpu_max_memory_mb"])
    source = pd.read_csv(cfg["pregenerated_data"]["train_with_rules_cot2_csv"])
    lengths = tokenizer(source["Prompt"].tolist(), truncation=False, padding=False)["input_ids"]
    longest = source.iloc[max(range(len(lengths)), key=lambda i: len(lengths[i]))]["Prompt"]
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"], bias="none", task_type="CAUSAL_LM",
        target_modules=find_all_linear_names(model)))
    model.config.use_cache = False
    model.train()
    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)
    observations = []
    selected = None
    for batch_size in (4, 2, 1):
        try:
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            encoded = tokenize_responses({"Prompt": [longest] * batch_size}, tokenizer,
                                          cfg["training"]["max_length"])
            features = [{key: value[i] for key, value in encoded.items()} for i in range(batch_size)]
            batch = {key: value.to("cuda") for key, value in collator(features).items()}
            start = time.monotonic()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss during the memory probe")
            loss.backward()
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_reserved() / 1024**2
            observation = {"batch_size": batch_size, "tokens_per_example": len(encoded["input_ids"][0]),
                           "peak_reserved_mb": round(peak_mb),
                           "forward_backward_seconds": round(time.monotonic() - start, 3)}
            observations.append(observation)
            print(json.dumps(observation), flush=True)
            if peak_mb <= 12000:
                selected = batch_size
                break
            del loss, batch
        except torch.cuda.OutOfMemoryError:
            observations.append({"batch_size": batch_size, "out_of_memory": True})
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
    if selected is None:
        raise RuntimeError("No tested microbatch left sufficient GPU memory headroom")
    cfg["training"]["per_device_train_batch_size"] = selected
    cfg["training"]["gradient_accumulation_steps"] = 8 // selected
    cfg["huggingface_token"] = "HF_TOKEN"
    destination = Path(cfg["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "config.used.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (destination / "batch_probe.json").write_text(json.dumps(observations, indent=2) + "\n", encoding="utf-8")
    print(f"Selected batch={selected}, accumulation={8 // selected}, effective batch=8", flush=True)


if __name__ == "__main__":
    main()

"""Reproduce the CPU export from the bundled GuidelineKG LoRA adapter.

This step is already done in the university handover. It needs Python 3.12,
torch, transformers, peft, accelerate and safetensors, and several GB of RAM.
HF_TOKEN is needed only if the gated base model is not already cached.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path


BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
BASE_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
PACKAGE = Path(__file__).resolve().parent.parent


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-original", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    adapter = PACKAGE / "adapter"
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, dtype=torch.float32,
        device_map={"": "cpu"}, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False, device_map={"": "cpu"})
    model.eval()
    load_seconds = time.perf_counter() - started
    print(f"Loaded base + GuidelineKG adapter on CPU in {load_seconds:.2f}s.", flush=True)
    baseline = None
    if args.benchmark_original:
        question = "What is heart failure?"
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                    repetition_penalty=1.3, no_repeat_ngram_size=3,
                                    pad_token_id=tokenizer.eos_token_id)
        elapsed = time.perf_counter() - started
        generated = output[0, inputs["input_ids"].shape[1]:]
        baseline = {"question": question, "seconds": elapsed, "output_tokens": len(generated),
                    "tokens_per_second": len(generated) / elapsed,
                    "answer": tokenizer.decode(generated, skip_special_tokens=True),
                    "runtime": "Transformers + PEFT, float32 CPU",
                    "threads": args.threads, "max_new_tokens": 128,
                    "decoding": {"do_sample": False, "repetition_penalty": 1.3, "no_repeat_ngram_size": 3}}
        (PACKAGE / "reports/original-cpu.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        print(f"Original runtime: {len(generated)} tokens in {elapsed:.2f}s ({len(generated)/elapsed:.2f} tok/s).", flush=True)
        del output, inputs, generated
    print("Merging the exact trained LoRA adapter into the base model...", flush=True)
    model = model.merge_and_unload(safe_merge=True)
    model = model.half()
    gc.collect()
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="3GB")
    tokenizer.save_pretrained(args.output)
    metadata = {"base_model": BASE_MODEL, "base_revision": BASE_REVISION,
                "adapter": "finetuned_LLaMA_3.2_1B_with_rules_CoT2",
                "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
                "merge": "PEFT merge_and_unload(safe_merge=True), merge in float32 then export float16",
                "weights": {p.name: sha256(p) for p in args.output.glob("*.safetensors")},
                "cpu_load_seconds": load_seconds, "original_benchmark": baseline}
    (PACKAGE / "reports/model-export.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Merged model saved to {args.output}.", flush=True)


if __name__ == "__main__":
    main()

"""Measure real streamed answers from the running application (standard library only)."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from urllib.request import Request, urlopen

QUESTIONS = ["What is heart failure?", "What does a Class I recommendation mean?", "What is the difference between HFrEF and HFpEF?"]
GAAPO_QUESTIONS = ["Create one patient-education scene explaining what heart failure is. Highlight Heart failure on screen.",
                   "Create one patient-education scene about asking your medical team for support when living with heart failure. Highlight Ask for support on screen."]


def ask(url, question, tokens=128, mode="plain"):
    messages = [{"role": "user", "content": question}]
    provenance = None
    if mode == "gaapo":
        artifact = json.loads((Path(__file__).resolve().parent.parent / "web/gaapo/prompt.json").read_text(encoding="utf-8"))
        messages.insert(0, {"role": "system", "content": artifact["system_prompt"]})
        provenance = artifact["provenance"]
    payload = {"model": "guidelinekg", "messages": messages,
               "temperature": 0, "max_tokens": tokens, "stream": True,
               "stream_options": {"include_usage": True}, "repeat_penalty": 1.3, "repeat_last_n": 2048}
    request = Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    first = None
    answer = ""
    usage = {}
    timings = {}
    finish_reason = None
    with urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            value = line[6:].strip()
            if value == b"[DONE]":
                break
            item = json.loads(value)
            if item.get("error"):
                raise RuntimeError(item["error"])
            usage = item.get("usage") or usage
            timings = item.get("timings") or timings
            for choice in item.get("choices", []):
                content = choice.get("delta", {}).get("content") or ""
                if content and first is None:
                    first = time.perf_counter() - started
                answer += content
                finish_reason = choice.get("finish_reason") or finish_reason
    elapsed = time.perf_counter() - started
    count = usage.get("completion_tokens") or timings.get("predicted_n")
    return {"question": question, "mode": mode, "gaapo_provenance": provenance, "elapsed_seconds": round(elapsed, 4),
            "first_text_seconds": round(first, 4) if first is not None else None,
            "output_tokens": count, "tokens_per_second": round(count / elapsed, 2) if count else None,
            "finish_reason": finish_reason, "answer": answer, "usage": usage, "server_timings": timings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "reports/cpu-benchmark.json")
    parser.add_argument("--label", default="CPU Q8_0, 4 threads")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--mode", choices=["plain", "gaapo"], default="plain")
    args = parser.parse_args()
    with urlopen(args.url + "/health", timeout=5) as response:
        assert response.status == 200
    records = []
    for question in GAAPO_QUESTIONS if args.mode == "gaapo" else QUESTIONS:
        record = ask(args.url, question, args.tokens, args.mode)
        records.append(record)
        print(f"{question}: first text {record['first_text_seconds']}s; {record['output_tokens']} tokens in {record['elapsed_seconds']}s; {record['tokens_per_second']} tok/s", flush=True)
    report = {"runtime": args.label, "platform": platform.platform(), "processor": platform.processor(),
              "max_tokens": args.tokens, "mode": args.mode, "questions": records,
              "note": "Actual local measurements, not a guarantee for other CPUs. These examples test operation, not clinical accuracy. Runtime and decoding differ from Transformers."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

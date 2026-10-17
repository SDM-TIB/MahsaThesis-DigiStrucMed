"""Optional ZeroGPU demo. The university's CPU package is the default runtime."""
import hashlib
import json
import os
from pathlib import Path
from queue import Empty
from threading import Event, Thread

import gradio as gr
import spaces
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT / "adapter"
if not ADAPTER.exists():
    ADAPTER = ROOT.parent / "adapter"
PROMPT_DIR = ROOT / "gaapo"
if not PROMPT_DIR.exists():
    PROMPT_DIR = ROOT.parent / "web/gaapo"
GAAPO = json.loads((PROMPT_DIR / "prompt.json").read_text(encoding="utf-8"))
if (GAAPO.get("schema_version") != 1
        or GAAPO["prompt_text"] != (PROMPT_DIR / "best_prompt.txt").read_text(encoding="utf-8").strip()
        or hashlib.sha256(GAAPO["prompt_text"].encode("utf-8")).hexdigest() != GAAPO["provenance"]["sha256"]
        or GAAPO["system_prompt"] != GAAPO["prompt_text"] + "\nOutput requirements: " + GAAPO["output_requirements"]):
    raise ValueError("The bundled GAAPO prompt does not match its provenance.")

# ZeroGPU requires placement at module scope so it can optimize CUDA transfers.
# A regular CPU run is available for validating this optional page locally.
DEVICE = "cuda" if os.environ.get("SPACES_ZERO_GPU", "").lower() in ("true", "1") or torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=REVISION, token=os.environ.get("HF_TOKEN"))
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=REVISION,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32, low_cpu_mem_usage=True,
    token=os.environ.get("HF_TOKEN"))
model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=False)
model = model.merge_and_unload(safe_merge=True).eval().to(DEVICE)


class Cancelled(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


@spaces.GPU(duration=60)
def answer(question, max_tokens, mode="gaapo"):
    question = (question or "").strip()
    if not question:
        raise gr.Error("Enter a question first.")
    if len(question) > 12000:
        raise gr.Error("Please use a shorter question.")
    if mode not in ("gaapo", "plain"):
        raise gr.Error("Select GAAPO best prompt or plain Q&A.")
    messages = [{"role": "user", "content": question}]
    if mode == "gaapo":
        messages.insert(0, {"role": "system", "content": GAAPO["system_prompt"]})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    if inputs["input_ids"].shape[1] + int(max_tokens) > 4096:
        raise gr.Error("Please use a shorter question.")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60)
    cancelled = Event()
    errors = []

    def generate():
        try:
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=int(max_tokens), do_sample=False,
                    repetition_penalty=1.3, no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id, streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([Cancelled(cancelled)]))
        except Exception as exception:
            errors.append(type(exception).__name__)
            streamer.end()

    worker = Thread(target=generate, daemon=True)
    worker.start()
    result = ""
    try:
        for text in streamer:
            result += text
            yield result
        if errors:
            raise gr.Error("Generation failed. Try a shorter question or wait for a new GPU allocation.")
    except Empty:
        raise gr.Error("Generation timed out. Please try again.")
    finally:
        cancelled.set()
        worker.join(timeout=5)


with gr.Blocks(title="GAAPO · GuidelineKG") as demo:
    gr.Markdown("# GAAPO · GuidelineKG\nUse the saved GAAPO best prompt for a storyboard, or ask a plain question. Each request is independent and uses one model generation.")
    mode = gr.Radio(choices=[("GAAPO best prompt · storyboard", "gaapo"), ("Plain GuidelineKG Q&A", "plain")],
                    value="gaapo", label="Mode")
    question = gr.Textbox(label="Your storyboard request or question", lines=4, max_length=12000)
    length = gr.Radio([128, 256, 512], value=256, label="Maximum answer tokens")
    with gr.Row():
        ask = gr.Button("Generate", variant="primary")
        stop = gr.Button("Stop")
    output = gr.Textbox(label="Model answer", lines=12, interactive=False, buttons=["copy"])
    submission = ask.click(answer, [question, length, mode], output, concurrency_limit=1)
    stop.click(fn=None, inputs=None, outputs=None, cancels=[submission])
    gr.Examples([["Create one patient-education scene explaining what heart failure is. Highlight Heart failure on screen.", "gaapo"],
                 ["What is heart failure?", "plain"]], [question, mode])
    with gr.Accordion("GAAPO-selected prompt and provenance", open=False):
        gr.Markdown(f"Run: `{GAAPO['provenance']['source_run']}` · Candidate: `{GAAPO['provenance']['candidate_id']}`\n\nPrompt SHA256: `{GAAPO['provenance']['sha256']}`")
        gr.Textbox(value=GAAPO["prompt_text"], label="Saved best prompt", lines=10, interactive=False)
        gr.Textbox(value=GAAPO["output_requirements"], label="Original target-model output requirements", lines=4, interactive=False)
    gr.Markdown("Research demonstration. Outputs have not been clinically validated. Built with Llama.")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=8).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), show_error=False)

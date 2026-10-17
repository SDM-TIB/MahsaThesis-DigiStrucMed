"""Streamlit interface for the GAAPO-optimized medical storyboard generator.

Two modes:
- Ask the GuidelineKG model directly (plain question, no schema -- its
  native rule-QA behavior).
- Generate a video-storyboard prompt from a natural-language request, using
  uploaded literature, followed by a source-grounded editing model.

Run with:
    streamlit run GAAPO/app.py
"""
from __future__ import annotations

import sys
import json
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

# `streamlit run GAAPO/app.py` puts this file's own directory (GAAPO/) on
# sys.path, not the repo root (unlike `python -m GAAPO.experiment`, which
# puts the cwd on sys.path) -- so BRINK, a sibling of GAAPO/, isn't
# importable unless the repo root is added explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

from BRINK.model_inference import generate_fixed, load_fixed_model
from GAAPO.models import configured_llm
from GAAPO.script_pipeline import extract_literature, run_pipeline, load_gaapo_prompt

load_dotenv(Path(__file__).parent / ".env", override=False)
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER = str(Path(__file__).resolve().parent.parent / "BRINK/model/guidelineKG/finetuned_LLaMA_3.2_1B_with_rules_CoT2")
REPO_ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="GAAPO Storyboard Generator", layout="wide")


@st.cache_resource(show_spinner="Loading GuidelineKG model (first time only)...")
def load_model(dtype_name: str, device: str):
    return load_fixed_model(BASE_MODEL, ADAPTER, dtype_name, device)


st.sidebar.header("Model settings")
device = st.sidebar.selectbox("Device", ["cuda:0", "cpu"], index=0)
dtype_name = st.sidebar.selectbox("Precision", ["float16", "float32"], index=0)
max_new_tokens = st.sidebar.slider("Draft output token budget", 256, 4096, 1536, step=256)

if st.sidebar.button("Load / reload model", type="primary"):
    st.session_state.tokenizer, st.session_state.model = load_model(dtype_name, device)
    st.sidebar.success("Model loaded.")

if "model" in st.session_state:
    st.sidebar.caption(f"Active: {Path(ADAPTER).name}")
else:
    st.sidebar.warning("No model loaded yet.")


def require_model():
    if "model" not in st.session_state:
        st.error("Load the model first from the sidebar.")
        st.stop()
    return st.session_state.tokenizer, st.session_state.model


st.title("GAAPO Storyboard Generator")
mode = st.radio(
    "What do you want to do?",
    ["Ask the GuidelineKG model a question", "Create an Animaker script (draft + editor)"],
    horizontal=True,
)

if mode == "Ask the GuidelineKG model a question":
    st.caption("Plain question, no schema -- the model's native GuidelineKG rule-QA behavior.")
    question = st.text_area("Your question")
    if st.button("Ask", type="primary") and question.strip():
        tokenizer, model = require_model()
        with st.spinner("Generating..."):
            answer = generate_fixed(tokenizer, model, [{"role": "user", "content": question}], max_new_tokens)
        st.markdown("**Answer:**")
        st.write(answer)

else:
    st.caption("GuidelineKG creates the medical draft; the editing stage uses the best prompt selected by GAAPO to prepare the Animaker script. Literature is optional.")
    prompt_path = st.text_input("GAAPO best_prompt.txt", value=str(REPO_ROOT / "GAAPO/runs/vm-real-test-02/best_prompt.txt"))
    try:
        selected_prompt, prompt_provenance = load_gaapo_prompt(prompt_path)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Cannot load the selected GAAPO prompt: {exc}")
        st.stop()
    with st.expander("GAAPO-selected prompt and provenance"):
        st.text(selected_prompt)
        st.json(prompt_provenance)
    st.info("The existing run selected this prompt for JSON generation by the medical model. Using it after a medical draft is a new experimental condition; its previous score does not evaluate this two-stage workflow.")
    request = st.text_area("Video request", height=240,
                           help="Specify language, audience, duration, required topics and visual style.")
    uploads = st.file_uploader("Optional literature", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    st.caption("The request, extracted literature and draft are sent to the configured external editing service when you generate.")
    with st.expander("Editing model settings"):
        editor_model = st.text_input("Editing model", value="openai/gpt-oss-120b")
        editor_url = st.text_input("API base URL", value="https://router.huggingface.co/v1")
        editor_key_env = st.text_input("API key environment variable", value="HF_TOKEN")
    if st.button("Generate Animaker script", type="primary"):
        st.session_state.pop("script_result", None)
        if not request.strip():
            st.error("Enter your video request first.")
            st.stop()
        tokenizer, model = require_model()
        try:
            sources = [extract_literature(f.name, f.getvalue()) for f in (uploads or [])]
            editor_settings = {"model": editor_model, "base_url": editor_url,
                               "api_key_env": editor_key_env}
            editor = configured_llm(editor_settings)
            from GAAPO.script_pipeline import draft_messages
            messages = draft_messages(request, sources)
            # Refuse oversized inputs rather than silently discarding literature.
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_tokens = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            context_limit = getattr(model.config, "max_position_embeddings", None)
            if context_limit and input_tokens + max_new_tokens > context_limit:
                raise ValueError("Literature exceeds the draft model context window. Upload relevant excerpts.")
            run_dir = REPO_ROOT / "GAAPO" / "runs" / ("script-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
            run_dir.mkdir(parents=True)
            metadata = {"target_base": BASE_MODEL, "target_adapter": ADAPTER,
                        "draft_max_new_tokens": max_new_tokens,
                        "decoding": {"do_sample": False, "repetition_penalty": 1.3, "no_repeat_ngram_size": 3},
                        "editor_settings": editor_settings, "input_tokens": input_tokens}
            def checkpoint(record):
                (run_dir / "pipeline.json").write_text(json.dumps({**metadata, **record}, ensure_ascii=False, indent=2), encoding="utf-8")
                if "draft" in record:
                    (run_dir / "draft.txt").write_text(record["draft"], encoding="utf-8")
                if "final_script" in record:
                    (run_dir / "final_script.txt").write_text(record["final_script"], encoding="utf-8")
            with st.spinner("Creating the GuidelineKG draft, then preparing the Animaker script..."):
                record = run_pipeline(request, sources,
                    lambda msgs: generate_fixed(tokenizer, model, msgs, max_new_tokens), editor, checkpoint,
                    gaapo_prompt=selected_prompt, gaapo_provenance=prompt_provenance)
            st.session_state.script_result = {"record": record, "directory": str(run_dir)}
        except Exception as exc:
            st.error(f"Script generation failed: {exc}")
            if "run_dir" in locals():
                st.caption(f"Available draft and progress saved in {run_dir}")
    if "script_result" in st.session_state:
        result = st.session_state.script_result
        st.subheader("Final Animaker script")
        st.text(result["record"]["final_script"])
        st.download_button("Download script", result["record"]["final_script"], "animaker_script.txt", "text/plain")
        with st.expander("Original fine-tuned model draft"):
            st.text(result["record"]["draft"])
        st.caption(f"Draft, final script, sources and settings saved in {result['directory']}")

"""BRINK chatbot — ask the GuidelineKG fine-tuned models anything, side by side.

Compares the two fine-tuned LLaMA-3.2-1B adapters on the same question:
  - "With rules (CoT2)"  — fine-tuned on rule-augmented CoT data
  - "Baseline (no rules)" — fine-tuned without rule context

Two ways to prompt, toggled in the sidebar:
  - Plain  : the raw question only (identical protocol to process_predictions.py
             / evaluate_brink.py, so answers match your evaluation).
  - Concept: prepend the domain description + relevant knowledge-graph facts for
             the entities in the question (reuses kg_concepts.build_concept_prompt).

Both adapters share one base model in memory (LoRA weights swapped with
set_adapter), so this holds a single copy of LLaMA-3.2-1B, not two.

Run:
    streamlit run brink_chat.py
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import streamlit as st
import torch
from dotenv import find_dotenv, load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

import kg_concepts

# Base model is gated on HuggingFace — load the token from .env so from_pretrained
# can authenticate (or use the already-downloaded cache).
load_dotenv(find_dotenv())

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTERS = {
    "With rules (CoT2)": "model/GuidelineKG/finetuned_LLaMA_3.2_1B_with_rules_CoT2",
    "Baseline (no rules)": "model/GuidelineKG/finetuned_LLaMA_3.2_1B_baseline",
}
BADGES = {"With rules (CoT2)": "🟢", "Baseline (no rules)": "⚪"}
FACTS_PATH = Path("data/GuidelineKG/facts.tsv")
DTYPES = {"float32 (CPU-safe)": torch.float32,
          "bfloat16": torch.bfloat16,
          "float16": torch.float16}

# The "domain concept": a system message telling the model what it was
# fine-tuned on, so it answers from the right frame of reference. Editable live
# in the sidebar; this is just the default. (Describes the GuidelineKG domain,
# unlike kg_concepts.DOMAIN_CONTEXT which describes the qKG questionnaire.)
DEFAULT_DOMAIN_PROMPT = (
    "You are a language model fine-tuned on a knowledge graph built from "
    "scientific medical guidelines. The graph encodes relationships between "
    "entities such as populations, patients, procedures, drugs, side effects, "
    "recommendations, levels of evidence, and the source documents they come "
    "from. You were fine-tuned with neuro-symbolic chain-of-thought rules mined "
    "from this graph, so you have internalized those relationships. Answer each "
    "question using this medical-guideline knowledge, directly and concisely."
)

st.set_page_config(page_title="BRINK Chat — GuidelineKG", layout="wide")


@st.cache_resource(show_spinner="Loading LLaMA-3.2-1B + both adapters (first time only)...")
def load_dual_model(dtype_name: str):
    """Load the base model once and attach every adapter to it."""
    names = list(ADAPTERS.keys())
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    # No device_map: a 1B model doesn't need accelerate's dispatch, and passing
    # device_map here makes PeftModel re-dispatch on adapter load — which, when
    # free RAM is low, tries to offload layers to disk and errors without an
    # offload_dir. A plain CPU load sidesteps that entirely.
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=DTYPES[dtype_name],
        low_cpu_mem_usage=True,
        token=token,
    )
    base.to("cpu")
    model = PeftModel.from_pretrained(base, ADAPTERS[names[0]], adapter_name=names[0])
    for name in names[1:]:
        model.load_adapter(ADAPTERS[name], adapter_name=name)
    model.eval()
    return tokenizer, model, names


@st.cache_data(show_spinner="Loading knowledge-graph facts...")
def load_facts_cached(path_str: str) -> kg_concepts.Facts:
    return kg_concepts.load_facts(Path(path_str))


def build_prompt(question: str, use_concept: bool, facts, entities: list[str] | None) -> str:
    if not use_concept:
        return question
    return kg_concepts.build_concept_prompt(question, facts, entities)


def generate_stream(tokenizer, model, adapter_name: str, prompt: str,
                    max_new_tokens: int, placeholder, system_prompt: str | None = None) -> str:
    """Switch to `adapter_name` and stream a greedy reply into `placeholder`."""
    model.set_adapter(adapter_name)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    text_in = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text_in, return_tensors="pt").to("cpu")
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    kwargs = dict(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id, streamer=streamer,
    )
    thread = threading.Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    text = ""
    for chunk in streamer:
        text += chunk
        placeholder.markdown(text + "▌")
    thread.join()
    text = text.strip()
    placeholder.markdown(text or "_(no answer)_")
    return text


# --- Sidebar -----------------------------------------------------------------

st.sidebar.header("Settings")
dtype_name = st.sidebar.selectbox("Precision", list(DTYPES.keys()), index=0)
max_new_tokens = st.sidebar.slider("Max new tokens (answer length)", 16, 2048, 256)

# The domain concept: a system prompt so the model knows what it was trained on.
use_domain = st.sidebar.toggle(
    "Domain prompt (tell the model what it knows)", value=True,
    help="Sent as a system message before every question, in both plain and "
         "concept modes. Edit the text below to change what the model is told.",
)
domain_prompt = ""
if use_domain:
    domain_prompt = st.sidebar.text_area(
        "Domain prompt", value=DEFAULT_DOMAIN_PROMPT, height=180,
        help="This is the 'you are fine-tuned on a medical-guidelines KG...' "
             "context. Change it freely and re-ask to see the effect.",
    )

use_concept = st.sidebar.toggle(
    "Concept context (prompt with KG facts)", value=False,
    help="Prepend the domain description + relevant knowledge-graph facts for the "
         "entities in your question. Off = ask the raw question (matches evaluation).",
)
facts_path_str = st.sidebar.text_input("Facts file", value=str(FACTS_PATH))
entity_hint = ""
if use_concept:
    entity_hint = st.sidebar.text_input(
        "Entity ids (optional, comma-separated)",
        help="KG entity ids the question is about. Leave blank to auto-detect "
             "from the question text.",
    )

missing = [p for p in ADAPTERS.values() if not Path(p, "adapter_config.json").is_file()]
if missing:
    st.sidebar.error("Adapters missing:\n" + "\n".join(missing))

if st.sidebar.button("Clear chat"):
    st.session_state.chat = []

st.sidebar.caption(
    "Both answers come from the same LLaMA-3.2-1B with different LoRA adapters. "
    "Running on CPU — a 1B reply takes a few seconds."
)


# --- Main: chat --------------------------------------------------------------

st.title("BRINK Chat — GuidelineKG")
st.caption("Ask a question; both fine-tuned models answer side by side.")

if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {question, concept, answers:{name:text}}

# Render history
for turn in st.session_state.chat:
    with st.chat_message("user"):
        st.markdown(turn["question"])
        if turn.get("concept"):
            st.caption("concept context used")
    with st.chat_message("assistant"):
        cols = st.columns(len(ADAPTERS))
        for col, name in zip(cols, ADAPTERS):
            with col:
                st.markdown(f"**{BADGES[name]} {name}**")
                st.markdown(turn["answers"].get(name, "_(no answer)_"))

question = st.chat_input("Ask the GuidelineKG models a question")
if question:
    if missing:
        st.error("Cannot run — some adapters are missing (see sidebar).")
        st.stop()

    facts = load_facts_cached(facts_path_str) if use_concept else None
    entities = (
        [e.strip() for e in entity_hint.split(",") if e.strip()]
        if (use_concept and entity_hint) else None
    )
    prompt = build_prompt(question, use_concept, facts, entities)

    with st.chat_message("user"):
        st.markdown(question)
        if use_concept:
            st.caption("concept context used")
            with st.expander("Prompt sent to the models"):
                st.code(prompt)

    tokenizer, model, names = load_dual_model(dtype_name)
    answers: dict[str, str] = {}
    with st.chat_message("assistant"):
        cols = st.columns(len(names))
        for col, name in zip(cols, names):
            with col:
                st.markdown(f"**{BADGES[name]} {name}**")
                placeholder = st.empty()
                with st.spinner(f"{name} is thinking..."):
                    answers[name] = generate_stream(
                        tokenizer, model, name, prompt, max_new_tokens, placeholder,
                        system_prompt=(domain_prompt or None),
                    )

    st.session_state.chat.append(
        {"question": question, "concept": use_concept, "answers": answers}
    )

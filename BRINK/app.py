"""Streamlit interface for the BRINK fine-tuned model.

Run with:
    streamlit run app.py

Provides:
- TSV upload -> gold JSON generation (reuses evaluation/create_gold.py parsing)
- Batch run of all gold questions through the fine-tuned model, with live
  streaming of each answer and a running Q/A log, optionally enriched with
  KG concept context (see kg_concepts.py) for prompt engineering
- Side-by-side comparison of the baseline and finetuned adapters on the
  same questions/prompts
- A chat tab to converse with the model directly
- JSON export for batch results, comparisons, and chat transcripts
"""

from __future__ import annotations

import ast
import csv
import io
import json
import threading
from pathlib import Path

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

import kg_concepts
from evaluation.create_gold import REQUIRED_COLUMNS, parse_answers

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTERS = {
    "With rules (CoT2)": "model/qKG/finetuned_LLaMA_3.2_1B_with_rules_CoT2",
    "Baseline (no rules)": "model/qKG/finetuned_LLaMA_3.2_1B_baseline",
}
DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
DEFAULT_BATCH_OUTPUT = Path("data/qKG/question/app_batch_predictions.json")
DEFAULT_CHAT_OUTPUT = Path("data/qKG/question/app_chat_log.json")
DEFAULT_COMPARE_OUTPUT = Path("data/qKG/question/app_compare_predictions.json")
FACTS_PATH = kg_concepts.DEFAULT_FACTS_PATH

st.set_page_config(page_title="BRINK Model Interface", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_model(adapter_path: str, dtype_name: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=DTYPES[dtype_name],
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading both adapters...")
def load_dual_model(dtype_name: str):
    """Load the base model once and attach every adapter to it.

    Adapters are LoRA weights on the same base model, so this lets us
    switch between "baseline" and "finetuned" with ``set_adapter`` instead
    of holding two full copies of the base model in memory.
    """
    names = list(ADAPTERS.keys())
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=DTYPES[dtype_name],
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTERS[names[0]], adapter_name=names[0])
    for name in names[1:]:
        model.load_adapter(ADAPTERS[name], adapter_name=name)
    model.eval()
    return tokenizer, model, names


@st.cache_data(show_spinner="Loading knowledge graph facts...")
def load_facts_cached(path_str: str) -> kg_concepts.Facts:
    return kg_concepts.load_facts(Path(path_str))


def generate_text(tokenizer, model, messages, max_new_tokens) -> str:
    """Non-streaming generation, used where two outputs are shown at once."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()


def build_prompt(question: str, use_concept: bool, facts: kg_concepts.Facts, entities: list[str] | None) -> str:
    if not use_concept:
        return question
    return kg_concepts.build_concept_prompt(question, facts, entities)


def generate_streaming(tokenizer, model, messages, max_new_tokens, placeholder=None) -> str:
    """Generate a reply token-by-token, updating ``placeholder`` as text arrives."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    text = ""
    for chunk in streamer:
        text += chunk
        if placeholder is not None:
            placeholder.markdown(text + "▌")
    thread.join()
    text = text.strip()
    if placeholder is not None:
        placeholder.markdown(text)
    return text


def parse_entity_cell(value: str) -> list[str] | None:
    """Parse an optional Python-list cell such as ``['217_M010_01']``."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, (list, tuple)):
        parsed = [parsed]
    entities = [str(item).strip() for item in parsed if str(item).strip()]
    return entities or None


def tsv_to_gold(raw_bytes: bytes) -> list[dict]:
    text = raw_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    columns = set(reader.fieldnames or [])
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(
            "Input TSV is missing required column(s): " + ", ".join(sorted(missing))
        )
    has_entities = "q_entity" in columns

    gold = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(reader, start=2):
        question_id = (row["id"] or "").strip()
        question = (row["question"] or "").strip()
        if not question_id:
            raise ValueError(f"Row {row_number}: id is empty")
        if question_id in seen_ids:
            raise ValueError(f"Row {row_number}: duplicate id {question_id!r}")
        if not question:
            raise ValueError(f"Row {row_number}: question is empty")

        answers = parse_answers(row["answer"] or "", row_number)
        entry = {
            "id": question_id,
            "question": question,
            "answers": answers,
            "hard_answer": answers[0],
        }
        if has_entities:
            entities = parse_entity_cell(row.get("q_entity", ""))
            if entities:
                entry["entities"] = entities
        gold.append(entry)
        seen_ids.add(question_id)
    return gold


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


# --- Sidebar: model settings -------------------------------------------------

st.sidebar.header("Model settings")
adapter_label = st.sidebar.selectbox("Adapter", list(ADAPTERS.keys()))
dtype_name = st.sidebar.selectbox("Precision", list(DTYPES.keys()), index=0)
max_new_tokens = st.sidebar.slider("Max new tokens", 16, 512, 128)

if st.sidebar.button("Load / reload model", type="primary"):
    st.session_state.tokenizer, st.session_state.model = load_model(
        ADAPTERS[adapter_label], dtype_name
    )
    st.session_state.loaded_adapter = adapter_label
    st.sidebar.success(f"Loaded: {adapter_label}")

if "model" in st.session_state:
    st.sidebar.caption(f"Active adapter: {st.session_state.loaded_adapter}")
else:
    st.sidebar.warning("No model loaded yet.")


def require_model():
    if "model" not in st.session_state:
        st.error("Load a model first from the sidebar.")
        st.stop()
    return st.session_state.tokenizer, st.session_state.model


# --- Tabs ---------------------------------------------------------------

tab_gold, tab_batch, tab_compare, tab_chat = st.tabs(
    ["1. Gold generation", "2. Batch run", "3. Compare baseline vs finetuned", "4. Chat"]
)

with tab_gold:
    st.subheader("Upload a question TSV to build the gold JSON")
    st.caption("Required columns: " + ", ".join(sorted(REQUIRED_COLUMNS)))
    st.caption(
        "Optional column: q_entity (a Python-list cell of KG entity ids, e.g. "
        "['217_M010_01']). When present it powers the concept-context prompt "
        "mode in the Batch run and Compare tabs."
    )
    uploaded_tsv = st.file_uploader("Question TSV", type=["tsv"])
    if uploaded_tsv is not None:
        try:
            gold = tsv_to_gold(uploaded_tsv.getvalue())
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state.gold = gold
            st.success(f"Parsed {len(gold)} questions.")
            st.dataframe(gold, width="stretch")
            st.download_button(
                "Download gold JSON",
                data=json.dumps(gold, ensure_ascii=False, indent=2),
                file_name="gold.json",
                mime="application/json",
            )

with tab_batch:
    st.subheader("Run every gold question through the model")
    gold_json_upload = st.file_uploader(
        "Or upload an existing gold JSON", type=["json"], key="gold_json_upload"
    )
    gold_data = None
    if gold_json_upload is not None:
        gold_data = json.loads(gold_json_upload.getvalue().decode("utf-8"))
    elif "gold" in st.session_state:
        gold_data = st.session_state.gold

    if gold_data is None:
        st.info("Generate or upload a gold JSON to enable batch running.")
    else:
        st.write(f"{len(gold_data)} question(s) loaded.")
        use_concept = st.checkbox(
            "Add concept context (prompt engineering using KG facts)",
            value=False,
            help=(
                "Prepends a domain description plus the question's semantic "
                "concept (category, questionnaire, answer meaning, ...) "
                "looked up from data/qKG/facts.tsv, instead of sending the "
                "bare question."
            ),
            key="batch_use_concept",
        )
        if use_concept:
            with st.expander("Preview prompt for the first question"):
                facts = load_facts_cached(str(FACTS_PATH))
                first = gold_data[0]
                st.code(
                    build_prompt(
                        first["question"], True, facts, first.get("entities")
                    )
                )
        output_path_str = st.text_input(
            "Save results to", value=str(DEFAULT_BATCH_OUTPUT)
        )
        run_clicked = st.button("Run all questions", type="primary")

        if run_clicked:
            tokenizer, model = require_model()
            facts = load_facts_cached(str(FACTS_PATH)) if use_concept else {}
            output_path = Path(output_path_str)
            results = []
            progress = st.progress(0.0)
            current_q = st.empty()
            current_a = st.empty()
            log_container = st.container()

            for index, item in enumerate(gold_data, start=1):
                question_id = str(item["id"])
                question = item["question"]
                prompt = build_prompt(question, use_concept, facts, item.get("entities"))
                current_q.markdown(f"**[{index}/{len(gold_data)}] Q:** {question}")
                answer = generate_streaming(
                    tokenizer,
                    model,
                    [{"role": "user", "content": prompt}],
                    max_new_tokens,
                    placeholder=current_a,
                )
                results.append(
                    {
                        "id": question_id,
                        "question": question,
                        "raw_output": answer,
                        "used_concept_context": use_concept,
                    }
                )
                save_json(output_path, results)
                progress.progress(index / len(gold_data))
                with log_container:
                    st.markdown(f"**{question_id}** — {question}")
                    st.markdown(f"> {answer}")
                    st.divider()

            st.session_state.batch_results = results
            st.success(f"Done. Saved {len(results)} results to {output_path}")

        if "batch_results" in st.session_state:
            st.download_button(
                "Download batch predictions JSON",
                data=json.dumps(
                    st.session_state.batch_results, ensure_ascii=False, indent=2
                ),
                file_name="batch_predictions.json",
                mime="application/json",
            )

with tab_compare:
    st.subheader("Run baseline and finetuned adapters side by side")
    st.caption(
        "Loads both adapters on top of one shared base model (memory "
        "efficient) and runs the same prompt through each, so you can "
        "compare their raw outputs directly."
    )
    compare_gold_upload = st.file_uploader(
        "Or upload an existing gold JSON", type=["json"], key="compare_gold_upload"
    )
    compare_gold_data = None
    if compare_gold_upload is not None:
        compare_gold_data = json.loads(compare_gold_upload.getvalue().decode("utf-8"))
    elif "gold" in st.session_state:
        compare_gold_data = st.session_state.gold

    if compare_gold_data is None:
        st.info("Generate or upload a gold JSON to enable comparison.")
    else:
        st.write(f"{len(compare_gold_data)} question(s) loaded.")
        compare_use_concept = st.checkbox(
            "Add concept context (prompt engineering using KG facts)",
            value=False,
            key="compare_use_concept",
        )
        compare_output_path_str = st.text_input(
            "Save results to", value=str(DEFAULT_COMPARE_OUTPUT)
        )
        compare_clicked = st.button("Run comparison", type="primary")

        if compare_clicked:
            tokenizer, dual_model, adapter_names = load_dual_model(dtype_name)
            facts = load_facts_cached(str(FACTS_PATH)) if compare_use_concept else {}
            compare_output_path = Path(compare_output_path_str)
            results = []
            progress = st.progress(0.0)
            log_container = st.container()

            for index, item in enumerate(compare_gold_data, start=1):
                question_id = str(item["id"])
                question = item["question"]
                gold_answer = item.get("hard_answer")
                prompt = build_prompt(
                    question, compare_use_concept, facts, item.get("entities")
                )
                messages = [{"role": "user", "content": prompt}]

                outputs = {}
                for name in adapter_names:
                    dual_model.set_adapter(name)
                    outputs[name] = generate_text(
                        tokenizer, dual_model, messages, max_new_tokens
                    )

                row = {
                    "id": question_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "used_concept_context": compare_use_concept,
                }
                for name, output in outputs.items():
                    row[name] = output
                results.append(row)
                save_json(compare_output_path, results)
                progress.progress(index / len(compare_gold_data))

                with log_container:
                    st.markdown(f"**{question_id}** — {question}")
                    if gold_answer is not None:
                        st.caption(f"Gold answer: {gold_answer}")
                    columns = st.columns(len(adapter_names))
                    for col, name in zip(columns, adapter_names):
                        with col:
                            st.markdown(f"*{name}*")
                            st.markdown(f"> {outputs[name]}")
                    st.divider()

            st.session_state.compare_results = results
            st.success(
                f"Done. Saved {len(results)} comparisons to {compare_output_path}"
            )

        if "compare_results" in st.session_state:
            st.dataframe(st.session_state.compare_results, width="stretch")
            st.download_button(
                "Download comparison JSON",
                data=json.dumps(
                    st.session_state.compare_results, ensure_ascii=False, indent=2
                ),
                file_name="compare_predictions.json",
                mime="application/json",
            )

with tab_chat:
    st.subheader("Chat with the fine-tuned model")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question")
    if user_input:
        tokenizer, model = require_model()
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            answer = generate_streaming(
                tokenizer,
                model,
                st.session_state.chat_history,
                max_new_tokens,
                placeholder=placeholder,
            )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )

    col_save, col_clear, col_download = st.columns(3)
    chat_output_path = Path(DEFAULT_CHAT_OUTPUT)
    if col_save.button("Save chat to JSON"):
        save_json(chat_output_path, st.session_state.chat_history)
        st.success(f"Saved to {chat_output_path}")
    if col_clear.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
    col_download.download_button(
        "Download chat JSON",
        data=json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=2),
        file_name="chat_log.json",
        mime="application/json",
        disabled=not st.session_state.chat_history,
    )

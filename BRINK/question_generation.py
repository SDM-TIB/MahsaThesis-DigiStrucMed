import os
import argparse
import random
import time
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tqdm.auto import tqdm

_ = load_dotenv(find_dotenv())

HF_MODEL = os.getenv("HF_MODEL", "openai/gpt-oss-120b")


def get_huggingface_client():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add a new Hugging Face token to the repository's "
            ".env file or export HF_TOKEN in the current shell."
        )
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )

import csv
import ast


def ensure_list_string(value):
    value = value.strip()
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return value  # 已经是 list 的字符串表示
    except Exception:
        pass
    return str([value]) if value else "[]"


def postprocess_entity_fields(input_path, output_path):
    with open(input_path, newline='', encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("Input file is missing header row.")

    for row in rows:
        for key in ["q_entity", "answer", "a_entity"]:
            row[key] = ensure_list_string(row.get(key, ""))

    with open(output_path, "w", newline='', encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Post-processed file saved to: {output_path}")


def safe_extract_first(item):
    try:
        parsed = ast.literal_eval(item)
        if isinstance(parsed, list) and parsed:
            return parsed[0]
    except:
        pass
    return item.strip()


def clean_output_entity_check(input_path, output_path):
    cleaned_rows = []
    with open(input_path, newline='', encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        headers = reader.fieldnames or []
        for row in reader:
            question = row.get("question", "").strip()

            # parse q_entity
            raw_q = row.get("q_entity", "").strip()
            try:
                q_list = ast.literal_eval(raw_q) if raw_q else []
            except Exception:
                q_list = []
            q_ent = q_list[0] if q_list else ""

            # parse a_entity
            raw_a = row.get("a_entity", "").strip()
            try:
                a_list = ast.literal_eval(raw_a) if raw_a else []
            except Exception:
                a_list = []
            a_ent = a_list[0] if a_list else ""

            if question and q_ent and q_ent in question and (not a_ent or a_ent not in question):
                cleaned_rows.append(row)

    with open(output_path, "w", newline='', encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(cleaned_rows)
    print(f"✅ Cleaned valid questions saved to: {output_path}")
    return len(cleaned_rows)


def clean_output(input_path, output_path):
    seen_ids = set()
    cleaned_rows = []
    with open(input_path, newline='', encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        headers = reader.fieldnames or []
        for row in reader:
            rec_id = row["id"].strip()
            if rec_id not in seen_ids:
                seen_ids.add(rec_id)
                cleaned_rows.append(row)
    with open(output_path, "w", newline='', encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(cleaned_rows)
    print(f"✅ Deduplicated by ID. {len(cleaned_rows)} unique rows saved to: {output_path}")
    return len(cleaned_rows)


def sort_by_id(input_path, output_path):
    """按ID从小到大排序文件"""
    with open(input_path, newline='', encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        rows = list(reader)
        headers = reader.fieldnames or []

    # 按ID排序（转换为整数进行排序）
    try:
        sorted_rows = sorted(rows, key=lambda x: int(x["id"]))
        print(f"✅ Sorted {len(sorted_rows)} rows by ID (integer)")
    except ValueError:
        # 如果ID不是整数，按字符串排序
        sorted_rows = sorted(rows, key=lambda x: x["id"])
        print(f"✅ Sorted {len(sorted_rows)} rows by ID (string)")

    with open(output_path, "w", newline='', encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"✅ Sorted file saved to: {output_path}")
    return len(sorted_rows)


def select_instances(df, n, method="top"):
    if method == "random":
        return df.sample(n, random_state=42)
    elif method == "top":
        if "support" in df.columns:
            return df.sort_values(by="support", ascending=False).head(n)
        else:
            return df.head(n)
    else:
        raise ValueError("Selection method must be 'random' or 'top'.")


def parse_grounding(grounding_str):
    if "=>" not in grounding_str:
        raise ValueError("Grounding string missing '=>' separator.")
    _, head_str = grounding_str.split("=>", 1)
    parts = head_str.strip().split()
    if len(parts) < 3:
        raise ValueError("Head grounding must have at least 3 tokens.")
    return {
        "entity_x": parts[0],
        "relation_T": parts[1],
        "entity_z": parts[2],
    }


def generate_qa_openai(
    instance,
    prompt_template,
    max_retries=8,
    retry_base_seconds=5.0,
    retry_max_seconds=120.0,
):
    client = get_huggingface_client()
    grounding_info = parse_grounding(instance["grounding"])
    prompt = prompt_template.format(
        entity_x=grounding_info["entity_x"],
        relation_T=grounding_info["relation_T"],
        entity_z=grounding_info["entity_z"],
    )
    transient_errors = (
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
    )
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=HF_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert KGQA question generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            break
        except transient_errors as exc:
            if attempt == max_retries:
                raise

            delay = min(retry_max_seconds, retry_base_seconds * (2 ** attempt))
            delay += random.uniform(0, min(1.0, delay * 0.1))
            tqdm.write(
                f"[RETRY] {type(exc).__name__}: API temporarily unavailable. "
                f"Retrying in {delay:.1f}s ({attempt + 1}/{max_retries})."
            )
            time.sleep(delay)
    answer_text = (response.choices[0].message.content or "").strip()

    qa = {"question": "", "q_entity": "", "a_entity": ""}
    for line in answer_text.splitlines():
        line = line.strip()
        if line.lower().startswith("question:"):
            qa["question"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("question entity:"):
            qa["q_entity"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("answer entity:"):
            qa["a_entity"] = line.split(":", 1)[1].strip()
    return qa


def get_prompt_template(prompt_type):
    if prompt_type == "general":
        return GENERAL_PROMPT_TEMPLATE
    else:
        raise ValueError("Unknown prompt type")


GENERAL_PROMPT_TEMPLATE = """
You are an expert in knowledge graph question generation.
A triple has been removed from a knowledge graph:

Removed Triple: ({entity_x}, {relation_T}, {entity_z})

Note: The numbers in the entities (e.g., {entity_x}, {entity_z}) represent specific individual identifiers.

Your task is to generate the following outputs:
1. Question Entity: Exactly one of the entities from the removed triple (either {entity_x} or {entity_z} exactly, with no additions).
2. Answer Entity: The remaining entity from the removed triple. That is, if you choose {entity_x} as the Question Entity, then the Answer Entity must be {entity_z}, and vice versa.
3. Question: A clear,  natural-language question asking for the Answer Entity from the removed triple. The question must include the given relation {relation_T} in a way that adheres to the following constraint:
   - While you may omit or adjust auxiliary parts of {relation_T} (for example, you may remove the suffix "_of"), the core word (the "xxx" in the "xxx_of" structure) must remain unchanged.
   - The relation {relation_T} should be expressed naturally in your question; you may paraphrase it instead of using the raw {relation_T}, as long as the core meaning (the "xxx" in "xxx_of") is preserved (for example, you can convert "born_in" to "was born in").


The question must include the Question Entity. The question must not include the Answer Entity.
The question must not be a simple yes/no question; it should require an explicit answer.


Please output exactly three lines, in plain text, without any Markdown formatting.
Do NOT use asterisks (*), bullets, numbering, or any other markup.

Example:
Removed Triple: ("Alice", "wife_of", "Carol").
Output:
Question: Who is Carol's wife?
Question Entity: Carol
Answer Entity: Alice

Now, generate the outputs for:
Removed Triple: ({entity_x}, {relation_T}, {entity_z})
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--n_instances", type=int, default=10)
    parser.add_argument("--model_choice", choices=["openai", "llama"], default="openai")
    parser.add_argument("--prompt_type", choices=["general", "family"], default="general")
    parser.add_argument("--max_retries", type=int, default=8)
    parser.add_argument("--retry_base_seconds", type=float, default=5.0)
    parser.add_argument("--retry_max_seconds", type=float, default=120.0)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, sep="\t")
    if "id" not in df.columns:
        raise ValueError("Input file must contain an 'id' column.")
    df["id"] = df["id"].astype(str)

    # Step 0: Clean duplicate IDs from input and save
    if df["id"].duplicated().any():
        num_dups = df["id"].duplicated().sum()
        print(f"[CLEAN] Found {num_dups} duplicate IDs. Keeping first occurrences.")
        df = df[~df["id"].duplicated(keep="first")]
        df.to_csv(args.input_file, sep="\t", index=False)

    # Step 1: Resume logic - load processed IDs
    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                processed_ids.add(row["id"])

    # Select the fixed target set first, then remove completed IDs. This keeps
    # a resumed run from generating n_instances additional records.
    df_target = select_instances(df, min(args.n_instances, len(df)))
    df_selected = df_target[~df_target["id"].isin(processed_ids)]
    if df_selected.empty:
        print(f"All {len(df_target)} requested instances already processed.")
        return

    print(
        f"Resuming with {len(processed_ids)} saved row(s); "
        f"{len(df_selected)} of {len(df_target)} requested row(s) remain."
    )
    prompt_template = get_prompt_template(args.prompt_type)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    write_header = not os.path.exists(args.output_file)

    if os.path.exists(args.output_file):
        print(f"Output file {args.output_file} exists. Cleaning it before appending new data.")
        clean_output(args.output_file, args.output_file)

    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                processed_ids.add(row["id"])

    with open(args.output_file, "a", newline='', encoding="utf-8") as fout:
        fieldnames = ["id", "rule_index", "question", "q_entity", "answer", "a_entity", "rule", "grounding"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        for idx, row in tqdm(df_selected.iterrows(), total=len(df_selected), desc="Generating QA"):
            inst = row.to_dict()

            # Required fields check
            for field in ["id", "rule_index", "rule", "grounding"]:
                if field not in inst:
                    raise KeyError(f"Missing required field: {field}")

            rec_id = inst["id"]
            if rec_id in processed_ids:
                continue

            if args.model_choice == "openai":
                qa = generate_qa_openai(
                    inst,
                    prompt_template,
                    max_retries=args.max_retries,
                    retry_base_seconds=args.retry_base_seconds,
                    retry_max_seconds=args.retry_max_seconds,
                )
            else:
                raise NotImplementedError("LLaMA not implemented.")

            question = qa["question"]
            q_ent = qa["q_entity"]
            a_ent = qa["a_entity"]

            if q_ent and q_ent not in question:
                print(f"[WARN] q_entity `{q_ent}` not in question: {question}")
                question = ""
            if a_ent and a_ent in question:
                print(f"[WARN] a_entity `{a_ent}` appears in question: {question}")
                question = ""

            writer.writerow({
                "id": rec_id,
                "rule_index": inst["rule_index"],
                "question": question,
                "q_entity": [q_ent] if q_ent else [],
                "answer": [a_ent] if a_ent else [],
                "a_entity": [a_ent] if a_ent else [],
                "rule": [inst["rule"]] if inst["rule"] else [],
                "grounding": [inst["grounding"]] if inst["grounding"] else []
            })

            # Preserve every completed API result for safe resume after an
            # interruption or exhausted retry sequence.
            fout.flush()
            processed_ids.add(rec_id)

    print(f"✅ QA dataset saved to {args.output_file}")

    # Step 4: Postprocess entity fields to ensure list format
    postprocess_entity_fields(args.output_file, args.output_file)
    print(f"✅ Postprocessed output saved to: {args.output_file}")

    # === Step 2: Always perform ID deduplication ===
    cleaned_output_path = (
        args.output_file if args.output_file.endswith("_cleaned.tsv")
        else args.output_file.replace(".tsv", "_cleaned.tsv")
    )

    clean_output(args.output_file, cleaned_output_path)
    print(f"✅ Cleaned output saved to {cleaned_output_path}")

    # ✅ Ensure all future steps use the cleaned version
    args.output_file = cleaned_output_path

    # Step 3: Remove invalid questions (question empty / q_ent not in question / a_ent in question)
    final_cleaned_path = cleaned_output_path.replace(".tsv", "_final.tsv")
    clean_output_entity_check(cleaned_output_path, final_cleaned_path)
    print(f"✅ Final cleaned output saved to: {final_cleaned_path}")

    # 🆕 Step 4: Sort by ID
    sort_by_id(final_cleaned_path, final_cleaned_path)
    print(f"🔢 Final output sorted by ID: {final_cleaned_path}")


if __name__ == "__main__":
    main()

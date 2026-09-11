"""
Read line26-example.json directly and export it as CSV files following
the schema in table_structure.sql (semantic_units / mentions /
mention_review_flags / mention_candidates / candidate_types).

No database is used - the JSON is flattened straight into CSVs.

Usage:
    python export_csv.py
"""

import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
JSON_PATH = BASE_DIR / "line26-example.json"
CSV_DIR = BASE_DIR / "csv"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, columns: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"Wrote {path}")


def build_tables(doc: dict):
    semantic_unit_id = doc["semantic_unit_id"]

    semantic_units_rows = [[
        semantic_unit_id,
        doc.get("schema_version"),
        doc.get("document_sha256"),
        doc.get("benchmark_category"),
        doc.get("export_unit_path"),
        doc.get("stage4_relpath"),
        doc.get("text_sha256"),
        doc.get("n_mentions"),
        doc.get("n_linked_mentions"),
        doc.get("model_name"),
        doc.get("linker_name"),
        doc.get("linker_k"),
        doc.get("linker_threshold"),
        doc.get("umls_source"),
        doc.get("label_status"),
        doc.get("gold_status"),
        doc.get("review_status"),
    ]]

    mentions_rows = []
    review_flags_rows = []
    candidates_rows = []
    candidate_types_rows = []

    for mention_index, mention in enumerate(doc.get("mentions", [])):
        mention_id = f"{semantic_unit_id}_m{mention_index}"

        mentions_rows.append([
            mention_id,
            semantic_unit_id,
            mention_index,
            mention.get("surface_form"),
            mention.get("char_start"),
            mention.get("char_end"),
            mention.get("ner_label"),
            mention.get("predicted_cui"),
            mention.get("predicted_score"),
            mention.get("predicted_canonical_name"),
            mention.get("label_status"),
            mention.get("gold_status"),
        ])

        for flag_index, flag in enumerate(mention.get("review_flags", [])):
            review_flags_rows.append([mention_id, flag_index, flag])

        for candidate_index, candidate in enumerate(mention.get("candidates", [])):
            candidate_id = f"{mention_id}_c{candidate_index}"

            candidates_rows.append([
                candidate_id,
                mention_id,
                candidate_index,
                candidate.get("cui"),
                candidate.get("score"),
                candidate.get("canonical_name"),
            ])

            for type_index, type_code in enumerate(candidate.get("types", [])):
                candidate_types_rows.append([candidate_id, type_index, type_code])

    return {
        "semantic_units": (
            ["semantic_unit_id", "schema_version", "document_sha256",
             "benchmark_category", "export_unit_path", "stage4_relpath",
             "text_sha256", "n_mentions", "n_linked_mentions", "model_name",
             "linker_name", "linker_k", "linker_threshold", "umls_source",
             "label_status", "gold_status", "review_status"],
            semantic_units_rows,
        ),
        "mentions": (
            ["mention_id", "semantic_unit_id", "mention_index", "surface_form",
             "char_start", "char_end", "ner_label", "predicted_cui",
             "predicted_score", "predicted_canonical_name", "label_status",
             "gold_status"],
            mentions_rows,
        ),
        "mention_review_flags": (
            ["mention_id", "flag_index", "flag"],
            review_flags_rows,
        ),
        "mention_candidates": (
            ["candidate_id", "mention_id", "candidate_index", "cui", "score",
             "canonical_name"],
            candidates_rows,
        ),
        "candidate_types": (
            ["candidate_id", "type_index", "type_code"],
            candidate_types_rows,
        ),
    }


def main() -> None:
    doc = load_json(JSON_PATH)
    tables = build_tables(doc)

    CSV_DIR.mkdir(exist_ok=True)
    for table_name, (columns, rows) in tables.items():
        write_csv(CSV_DIR / f"{table_name}.csv", columns, rows)


if __name__ == "__main__":
    main()

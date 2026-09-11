"""Convert a BRINK question TSV file into the evaluation gold JSON format."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "data/qKG/question/val_cleaned_final.tsv"
)
DEFAULT_OUTPUT = Path(
    "data/qKG/question/val_cleaned_final_gold.json"
)
REQUIRED_COLUMNS = {"id", "question", "answer"}


def parse_answers(value: str, row_number: int) -> list[str]:
    """Parse a Python-list answer cell such as ``['Paris', 'London']``."""
    value = value.strip()
    if not value:
        raise ValueError(f"Row {row_number}: the answer field is empty")

    try:
        parsed: Any = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Row {row_number}: invalid answer list {value!r}"
        ) from exc

    if isinstance(parsed, (list, tuple, set)):
        answers = [str(answer).strip() for answer in parsed]
    else:
        answers = [str(parsed).strip()]

    answers = [answer for answer in answers if answer]
    if not answers:
        raise ValueError(f"Row {row_number}: no non-empty answers were found")
    return answers


def create_gold(input_path: Path, output_path: Path) -> int:
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                "Input TSV is missing required column(s): "
                + ", ".join(sorted(missing))
            )

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
            gold.append(
                {
                    "id": question_id,
                    "question": question,
                    "answers": answers,
                    "hard_answer": answers[0],
                }
            )
            seen_ids.add(question_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(gold, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return len(gold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create BRINK gold-answer JSON from a question TSV file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Question TSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Gold JSON path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = create_gold(args.input, args.output)
    print(f"Created {count} gold examples: {args.output}")


if __name__ == "__main__":
    main()

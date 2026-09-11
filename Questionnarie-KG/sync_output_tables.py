"""Create/update the curated tables in output_tables/ from the current
Source/ workbooks (DigiLearnHF_evaluation_sosci.xlsx + codebook).

output_tables/ extends the plain RDFizer input tables (see
export_dbml_tables.py) with a few hand-curated columns that cannot be
derived from the source data:

- Questions.csv: QuestionBodyEnglish, questionCategory
- Answer.csv:    ResponseLabelEnglish, answerCategory
- Answer.csv:    the codebook's explicit "-9 / not answered" option is
                  retyped to Type=NullAnswer and used as the single
                  "no response" answer for its question (instead of also
                  keeping a separate synthetic "no recorded response" row).

Re-running this script keeps output_tables/ in sync with Source/ while
preserving every hand-curated value already on disk:

- Existing (QuestionId, ResponseCode) / QuestionId rows keep their prior
  AnswerId, Type, English text and category, and only get their
  source-derived columns (German text, RawNumber, ...) refreshed.
- Brand new rows get a best-effort default (translation placeholder,
  heuristic category for a clean 1..N rating scale) and are called out in
  the sync report so a human can review them.
- Rows that disappeared from the source are dropped and listed in the
  report so nothing is silently lost.

Usage:
    python sync_output_tables.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd

import export_dbml_tables as base

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output_tables"

NEEDS_TRANSLATION_PREFIX = "[NEEDS TRANSLATION] "
NEEDS_REVIEW = "NEEDS_REVIEW"
ANSWER_CATEGORY_SCALE = ["very_negative", "negative", "neutral", "positive", "very_positive"]

QUESTION_COLUMNS = [
    "QuestionId",
    "QuestionBody",
    "QuestionBodyEnglish",
    "QuestionnaireId",
    "questionCategory",
]
ANSWER_COLUMNS = [
    "AnswerId",
    "QuestionId",
    "ResponseCode",
    "ResponseLabel",
    "ResponseLabelEnglish",
    "Type",
    "RawNumber",
    "answerCategory",
]


def read_prior_table(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    try:
        # keep_default_na=False: cells like the literal ResponseCode "null"
        # must stay as text, not become an actual NaN and break key matching.
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[])
    except pd.errors.ParserError as error:
        # A hand-edited/corrupted prior file (e.g. mismatched CSV quoting)
        # shouldn't crash the whole sync; treat it as if nothing existed yet,
        # so every row is regenerated fresh (with translation/category
        # placeholders flagged in the report for review).
        print(f"WARNING: {path.name} could not be parsed ({error}); treating it as empty.")
        return pd.DataFrame()


def is_blank(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == ""


def suggest_answer_categories(codes_with_numbers: list[tuple[str, int | None]]) -> dict[str, str]:
    """Suggest a 5-bucket category (very_negative..very_positive) for a
    question's scale options, only when they form a clean 1..N ladder.
    Anything else is left for manual review."""
    if not codes_with_numbers or any(n is None for _, n in codes_with_numbers):
        return {code: NEEDS_REVIEW for code, _ in codes_with_numbers}

    numbers = sorted(n for _, n in codes_with_numbers)
    n = len(numbers)
    if numbers != list(range(1, n + 1)):
        return {code: NEEDS_REVIEW for code, _ in codes_with_numbers}

    result = {}
    for code, number in codes_with_numbers:
        bucket = min(max(math.ceil(number * 5 / n), 1), 5)
        result[code] = ANSWER_CATEGORY_SCALE[bucket - 1]
    return result


def sync_platform(fresh_platform: pd.DataFrame) -> pd.DataFrame:
    return fresh_platform.copy()


def sync_user(fresh_user: pd.DataFrame, prior_user: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {"new_users": [], "removed_users": []}
    fresh_ids = set(fresh_user["UserId"].astype(str))
    if not prior_user.empty:
        prior_ids = set(prior_user["UserId"].astype(str))
        report["new_users"] = sorted(fresh_ids - prior_ids, key=int)
        report["removed_users"] = sorted(prior_ids - fresh_ids, key=int)
    return fresh_user.copy(), report


def sync_questionnaire(fresh_questionnaire: pd.DataFrame) -> pd.DataFrame:
    return fresh_questionnaire.rename(
        columns={"QuestionnarieId": "QuestionnaireId", "QuestionnarieType": "QuestionnaireType"}
    )


def sync_questions(fresh_question: pd.DataFrame, prior_question: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fresh = fresh_question.rename(columns={"QuestionnarieId": "QuestionnaireId"})
    prior_by_id = (
        {row.QuestionId: row for row in prior_question.itertuples(index=False)}
        if not prior_question.empty
        else {}
    )

    report: dict = {"new_questions": [], "removed_questions": [], "changed_text": []}
    rows = []
    for row in fresh.itertuples(index=False):
        prior_row = prior_by_id.get(row.QuestionId)
        if prior_row is not None:
            english = prior_row.QuestionBodyEnglish
            category = prior_row.questionCategory
            if str(getattr(prior_row, "QuestionBody", "")).strip() != str(row.QuestionBody).strip():
                report["changed_text"].append(
                    (row.QuestionId, getattr(prior_row, "QuestionBody", ""), row.QuestionBody)
                )
        else:
            english = NEEDS_TRANSLATION_PREFIX + str(row.QuestionBody)
            category = NEEDS_REVIEW
            report["new_questions"].append(row.QuestionId)

        rows.append(
            {
                "QuestionId": row.QuestionId,
                "QuestionBody": row.QuestionBody,
                "QuestionBodyEnglish": english,
                "QuestionnaireId": row.QuestionnaireId,
                "questionCategory": category,
            }
        )

    if prior_by_id:
        fresh_ids = set(fresh["QuestionId"])
        report["removed_questions"] = sorted(set(prior_by_id) - fresh_ids)

    return pd.DataFrame(rows, columns=QUESTION_COLUMNS), report


def collapse_null_answers(fresh_answer: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    """For each question with an explicit "-9" codebook option, retype it
    as the question's NullAnswer and drop the synthetic no-recorded-response
    row, remapping its AnswerId to the "-9" row's AnswerId."""
    remap: dict[int, int] = {}
    kept_groups = []

    for question_id, group in fresh_answer.groupby("QuestionId", sort=False):
        minus9_rows = group[group["ResponseCode"] == "-9"]
        null_rows = group[group["Type"] == "NullAnswer"]

        if len(minus9_rows) == 1 and len(null_rows) == 1:
            minus9_answer_id = int(minus9_rows.iloc[0]["AnswerId"])
            null_answer_id = int(null_rows.iloc[0]["AnswerId"])
            remap[null_answer_id] = minus9_answer_id

            group = group.drop(index=null_rows.index).copy()
            group.loc[group["ResponseCode"] == "-9", "Type"] = "NullAnswer"

        kept_groups.append(group)

    collapsed = pd.concat(kept_groups, ignore_index=True) if kept_groups else fresh_answer
    return collapsed, remap


def sync_answers(
    fresh_answer: pd.DataFrame, prior_answer: pd.DataFrame
) -> tuple[pd.DataFrame, dict[int, int], dict]:
    collapsed, remap = collapse_null_answers(fresh_answer)

    def raw_number(code: object) -> int | None:
        try:
            return int(str(code))
        except (TypeError, ValueError):
            return None

    collapsed = collapsed.copy()
    # Nullable Int64 (not float64) so blank RawNumber values don't force the
    # whole column to render as "1.0" instead of "1" in the CSV.
    collapsed["RawNumber"] = collapsed["ResponseCode"].apply(raw_number).astype("Int64")

    prior_by_key = (
        {(row.QuestionId, str(row.ResponseCode)): row for row in prior_answer.itertuples(index=False)}
        if not prior_answer.empty
        else {}
    )

    report: dict = {
        "new_answers": [],
        "removed_answers": [],
        "changed_text": [],
        "needs_category_review": [],
    }

    result_rows = []
    fresh_keys: set[tuple[str, str]] = set()

    for question_id, group in collapsed.groupby("QuestionId", sort=False):
        scale_candidates = [
            (r.ResponseCode, r.RawNumber) for r in group.itertuples(index=False) if r.Type not in ("NullAnswer", "Textual")
        ]
        suggested = suggest_answer_categories(scale_candidates)

        for r in group.itertuples(index=False):
            key = (question_id, str(r.ResponseCode))
            fresh_keys.add(key)
            prior_row = prior_by_key.get(key)

            if prior_row is not None:
                answer_id = prior_row.AnswerId
                answer_type = prior_row.Type
                english = prior_row.ResponseLabelEnglish
                category = prior_row.answerCategory
                if str(prior_row.ResponseLabel).strip() != str(r.ResponseLabel).strip():
                    report["changed_text"].append((question_id, r.ResponseCode, prior_row.ResponseLabel, r.ResponseLabel))
            else:
                answer_id = r.AnswerId
                answer_type = r.Type
                report["new_answers"].append((question_id, r.ResponseCode))
                if answer_type == "NullAnswer":
                    english = "No recorded response"
                    category = "Empty"
                elif answer_type == "Textual":
                    english = "Textual response"
                    category = "textual"
                else:
                    english = NEEDS_TRANSLATION_PREFIX + str(r.ResponseLabel)
                    category = suggested.get(r.ResponseCode, NEEDS_REVIEW)
                    if category == NEEDS_REVIEW:
                        report["needs_category_review"].append((question_id, r.ResponseCode))

            result_rows.append(
                {
                    "AnswerId": answer_id,
                    "QuestionId": question_id,
                    "ResponseCode": r.ResponseCode,
                    "ResponseLabel": r.ResponseLabel,
                    "ResponseLabelEnglish": english,
                    "Type": answer_type,
                    "RawNumber": r.RawNumber,
                    "answerCategory": category,
                }
            )

    if prior_by_key:
        report["removed_answers"] = sorted(set(prior_by_key) - fresh_keys)

    return pd.DataFrame(result_rows, columns=ANSWER_COLUMNS), remap, report


def sync_response(fresh_response: pd.DataFrame, remap: dict[int, int]) -> pd.DataFrame:
    df = fresh_response.copy()
    df["AnswerId"] = df["AnswerId"].apply(lambda value: remap.get(int(value), int(value)))
    return df


def print_report(name: str, report: dict) -> None:
    lines = []
    for key, values in report.items():
        if values:
            lines.append(f"  {key}: {len(values)}")
            for value in values[:10]:
                lines.append(f"    - {value}")
            if len(values) > 10:
                lines.append(f"    ... and {len(values) - 10} more")
    if lines:
        print(f"[{name}]")
        print("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fresh = base.build_tables()

    prior_user = read_prior_table("User.csv")
    prior_question = read_prior_table("Questions.csv")
    prior_answer = read_prior_table("Answer.csv")

    platform_out = sync_platform(fresh["platform"])
    user_out, user_report = sync_user(fresh["user"], prior_user)
    questionnaire_out = sync_questionnaire(fresh["questionnaire"])
    question_out, question_report = sync_questions(fresh["question"], prior_question)
    answer_out, remap, answer_report = sync_answers(fresh["answer"], prior_answer)
    response_out = sync_response(fresh["response"], remap)

    # Match the quoting convention already on disk for each file so re-runs
    # produce minimal, meaningful diffs.
    platform_out.to_csv(OUTPUT_DIR / "Platform.csv", index=False, encoding="utf-8")
    user_out.to_csv(OUTPUT_DIR / "User.csv", index=False, encoding="utf-8")
    questionnaire_out.to_csv(OUTPUT_DIR / "Questionnaire.csv", index=False, encoding="utf-8")
    question_out.to_csv(OUTPUT_DIR / "Questions.csv", index=False, encoding="utf-8")
    answer_out.to_csv(
        OUTPUT_DIR / "Answer.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL
    )
    response_out.to_csv(
        OUTPUT_DIR / "Response.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL
    )

    obsolete_file = OUTPUT_DIR / "Answered.csv"
    if obsolete_file.exists():
        obsolete_file.unlink()

    print("output_tables sync complete.")
    print(f"Output folder: {OUTPUT_DIR}")
    print(
        "Row counts -> "
        f"Platform: {len(platform_out)}, User: {len(user_out)}, "
        f"Questionnaire: {len(questionnaire_out)}, Questions: {len(question_out)}, "
        f"Answer: {len(answer_out)}, Response: {len(response_out)}"
    )
    print()
    print("Sync report (review NEEDS_TRANSLATION / NEEDS_REVIEW placeholders before publishing):")
    print_report("User", user_report)
    print_report("Questions", question_report)
    print_report("Answer", answer_report)


if __name__ == "__main__":
    main()

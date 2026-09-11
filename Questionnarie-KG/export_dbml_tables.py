from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "Source"
OUTPUT_DIR = BASE_DIR / "input-tables"

EVALUATION_FILE = SOURCE_DIR / "DigiLearnHF_evaluation_sosci.xlsx"
CODEBOOK_FILE = SOURCE_DIR / "codebook_DigiLearnHF_2025-09-11_12-21.xlsx"

# Non-response columns SoSci always puts before the actual question
# variables in a codebook / export. Used to line up a combined workbook's
# data-sheet columns with the codebook's canonical variable order.
CODEBOOK_LEADING_COLUMNS = {"CASE", "SERIAL", "REF", "QUESTNNR", "MODE", "STARTED"}
CODEBOOK_REQUIRED_COLUMNS = {
    "Variable",
    "Variable Label",
    "Response Code",
    "Response Label",
    "Variable Type",
}

# Literal placeholder text that shows up in some codebook exports where a
# real tab character should be (e.g. "2<tab>Some label"). Source files are
# not always consistent about this, so it is normalized away on read.
TAB_PLACEHOLDER_RE = re.compile(r"<tab>", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")

QUESTIONNAIRE_NAME_BY_ID = {
    "SUS": "System Usability Scale",
    "uMARS": "Mobile Anwendungen Rating Skala",
    "ME1": "Module Evaluation1",
}
QUESTIONNAIRE_TYPE_BY_ID = {
    "SUS": "SUS",
    "uMARS": "uMARS",
    "ME1": "ModuleEvaluation",
}
NO_RECORDED_RESPONSE_VALUE = "NO_RECORDED_RESPONSE"


def normalize_case_id(value: object) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"\d+", text)
    if not match:
        return None
    return int(match.group(0))


def normalize_questnnr(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"M\d+", text, flags=re.IGNORECASE):
        return "ME1"
    if text.lower() == "umars":
        return "uMARS"
    if text.upper() == "SUS":
        return "SUS"
    return text


def questionnaire_for_var(var_name: str) -> str | None:
    if re.fullmatch(r"SU\d{2}_\d{2}", var_name):
        return "SUS"
    if re.fullmatch(r"UM\d{2}(?:_\d{2})?", var_name):
        return "uMARS"
    if re.fullmatch(r"M\d{3}(?:_\d{2})?", var_name):
        return "ME1"
    return None


def parse_question_components(var_name: str) -> tuple[int, int]:
    """Return a stable (family_code, question_code) pair from variable id."""
    m = re.fullmatch(r"M(\d{3})(?:_(\d{2}))?", var_name)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2) or "0")
        return 1, major * 100 + minor

    m = re.fullmatch(r"SU(\d{2})_(\d{2})", var_name)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2))
        return 2, major * 100 + minor

    m = re.fullmatch(r"UM(\d{2})(?:_(\d{2}))?", var_name)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2) or "0")
        return 3, major * 100 + minor

    # Fallback for unexpected ids; still deterministic.
    return 9, abs(hash(var_name)) % 100_000


def normalize_response_code(value: object) -> str:
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0", text):
        return text[:-2]
    return text


def answer_type_for_variable(variable_type: object) -> str:
    normalized_type = str(variable_type).strip().upper()
    return {
        "NOMINAL": "Nominal",
        "ORDINAL": "Ordinal",
        "TEXT": "Textual",
    }.get(normalized_type, normalized_type.title())


def complete_sus_likert_options(codebook_answers: pd.DataFrame) -> pd.DataFrame:
    """Add SUS Likert response codes that are present in the data but absent from the codebook."""
    rows_to_add: list[dict[str, object]] = []
    sus_questions = codebook_answers[
        codebook_answers["VAR"].astype(str).str.fullmatch(r"SU\d{2}_\d{2}")
    ]

    for question_id, question_answers in sus_questions.groupby("VAR", sort=False):
        existing_codes = {
            normalize_response_code(value)
            for value in question_answers["RESPONSE"].dropna().tolist()
        }
        template = question_answers.iloc[0].to_dict()
        for response_code in ("1", "2", "3", "4", "5"):
            if response_code in existing_codes:
                continue
            row = template.copy()
            row["RESPONSE"] = response_code
            row["MEANING"] = f"Likert response {response_code}"
            rows_to_add.append(row)

    if rows_to_add:
        codebook_answers = pd.concat(
            [codebook_answers, pd.DataFrame(rows_to_add)],
            ignore_index=True,
        )

    codebook_answers = codebook_answers.copy()
    codebook_answers["_Source_Order"] = range(len(codebook_answers))
    codebook_answers["_Response_Code"] = codebook_answers["RESPONSE"].apply(normalize_response_code)
    codebook_answers["_Sus_Order"] = pd.NA
    sus_order = {str(value): value for value in range(1, 6)}
    sus_order["-9"] = 6
    sus_mask = codebook_answers["VAR"].astype(str).str.fullmatch(r"SU\d{2}_\d{2}")
    codebook_answers.loc[sus_mask, "_Sus_Order"] = (
        codebook_answers.loc[sus_mask, "_Response_Code"].map(sus_order).fillna(99)
    )

    return (
        codebook_answers.sort_values(
            ["VAR", "_Sus_Order", "_Source_Order"],
            na_position="last",
            kind="stable",
        )
        .drop(columns=["_Source_Order", "_Response_Code", "_Sus_Order"])
        .reset_index(drop=True)
    )


def clean_label_text(value: object) -> object:
    """Normalize source text: drop the literal "<tab>" placeholder some
    codebook exports contain, and collapse repeated whitespace."""
    if pd.isna(value):
        return value
    text = TAB_PLACEHOLDER_RE.sub(" ", str(value))
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def resolve_existing_file(path: Path) -> Path:
    """Resolve a source file, falling back to BASE_DIR and common extensions
    if it is not found at ``path`` (e.g. Source/) as given."""
    candidates = [
        path,
        path.with_suffix(".xlsx"),
        path.with_suffix(".xls"),
        path.with_suffix(".csv"),
        BASE_DIR / path.name,
        BASE_DIR / path.with_suffix(".xlsx").name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    expected_names = ", ".join(dict.fromkeys(candidate.name for candidate in candidates))
    raise FileNotFoundError(
        f"{path.name} not found. Expected one of: {expected_names}"
    )


def resolve_existing_codebook(path: Path) -> Path:
    return resolve_existing_file(path)


def read_codebook(path: Path) -> pd.DataFrame:
    path = resolve_existing_codebook(path)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        codebook_df = pd.read_excel(path, sheet_name=0, dtype=str)
    elif path.suffix.lower() == ".csv":
        codebook_df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported codebook file type: {path.suffix}")

    codebook_df.columns = codebook_df.columns.str.strip()
    validate_codebook_columns(codebook_df, path.name)
    return codebook_df


def validate_codebook_columns(codebook_df: pd.DataFrame, file_label: str) -> None:
    missing_columns = sorted(CODEBOOK_REQUIRED_COLUMNS - set(codebook_df.columns))
    if missing_columns:
        raise ValueError(
            f"{file_label} is missing required columns: {', '.join(missing_columns)}"
        )


def find_combined_workbook() -> Path | None:
    """Look for a single workbook that holds both the respondent data and
    the codebook as two sheets (data + a sheet shaped like a codebook).
    Returns None if no such file is present, so the classic two-file
    Source/ layout keeps working unchanged."""
    if not SOURCE_DIR.exists():
        return None

    for path in sorted(SOURCE_DIR.glob("*.xlsx")):
        try:
            workbook = pd.ExcelFile(path)
        except (ValueError, OSError):
            continue
        if len(workbook.sheet_names) != 2:
            continue

        for sheet_name in workbook.sheet_names:
            header = pd.read_excel(path, sheet_name=sheet_name, nrows=0)
            columns = {str(column).strip() for column in header.columns}
            if CODEBOOK_REQUIRED_COLUMNS <= columns:
                return path

    return None


def load_combined_source(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a combined workbook (data sheet + codebook sheet) into the same
    (raw_eval, codebook_df) shape produced by the classic two-file layout.

    The data sheet in this format uses human-friendly column names instead
    of the codebook's Variable ids and has no CASE id, so that is
    reconstructed here:
    - Data columns are renamed back to Variable ids using the codebook's own
      variable order (skipping the leading CASE/SERIAL/.../STARTED columns,
      which this data sheet doesn't carry).
    - CASE is synthesized as a 1-based row sequence.
    """
    workbook = pd.ExcelFile(path)

    codebook_sheet = None
    for sheet_name in workbook.sheet_names:
        header = pd.read_excel(path, sheet_name=sheet_name, nrows=0)
        columns = {str(column).strip() for column in header.columns}
        if CODEBOOK_REQUIRED_COLUMNS <= columns:
            codebook_sheet = sheet_name
            break
    if codebook_sheet is None:
        raise ValueError(f"{path.name}: no sheet with the expected codebook columns was found")

    data_sheets = [name for name in workbook.sheet_names if name != codebook_sheet]
    data_sheet = data_sheets[0]

    codebook_df = pd.read_excel(path, sheet_name=codebook_sheet, dtype=str)
    codebook_df.columns = codebook_df.columns.str.strip()
    validate_codebook_columns(codebook_df, f"{path.name} [{codebook_sheet}]")

    data_df = pd.read_excel(path, sheet_name=data_sheet)

    canonical_order = list(dict.fromkeys(codebook_df["Variable"].dropna().tolist()))
    canonical_question_vars = [v for v in canonical_order if v not in CODEBOOK_LEADING_COLUMNS]
    data_columns = [c for c in data_df.columns if c not in CODEBOOK_LEADING_COLUMNS]

    if len(canonical_question_vars) < len(data_columns):
        raise ValueError(
            f"{path.name}: {len(data_columns)} data columns in '{data_sheet}' but only "
            f"{len(canonical_question_vars)} question variables in '{codebook_sheet}' to map them to."
        )

    rename_map = dict(zip(data_columns, canonical_question_vars[: len(data_columns)]))
    raw_eval = data_df.rename(columns=rename_map)

    if "CASE" not in raw_eval.columns:
        raw_eval.insert(0, "CASE", range(1, len(raw_eval) + 1))

    # build_tables() treats row 0 as a metadata/label row and drops it.
    meta_row = pd.DataFrame([{column: pd.NA for column in raw_eval.columns}])
    raw_eval = pd.concat([meta_row, raw_eval], ignore_index=True)

    return raw_eval, codebook_df


def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (raw_eval, codebook_df), preferring a combined workbook in
    Source/ if one is present, otherwise the classic two separate files."""
    combined_path = find_combined_workbook()
    if combined_path is not None:
        return load_combined_source(combined_path)

    evaluation_file = resolve_existing_file(EVALUATION_FILE)
    raw_eval = pd.read_excel(evaluation_file, sheet_name=0)
    codebook_df = read_codebook(CODEBOOK_FILE)
    return raw_eval, codebook_df


def build_tables() -> dict[str, pd.DataFrame]:
    """Build the RDFizer input tables from the current Source/ workbooks.

    Returns a dict of export-ready DataFrames keyed by table name
    (platform, user, questionnaire, question, answer, response), using the
    same column names written to input-tables/*.csv.
    """
    raw_eval, codebook_df = load_source_data()

    # Row 0 stores labels/sections, not respondent data.
    eval_df = raw_eval.iloc[1:].copy()

    # Extract variables and values from main DigiLearnHF sheet
    variables_df = codebook_df[["Variable", "Variable Label", "Variable Type"]].copy()
    variables_df.columns = ["VAR", "LABEL", "TYPE"]
    variables_df["LABEL"] = variables_df["LABEL"].apply(clean_label_text)

    # Extract response codes and labels
    values_df = codebook_df[["Variable", "Response Code", "Response Label"]].copy()
    values_df.columns = ["VAR", "RESPONSE", "MEANING"]
    values_df = values_df.dropna(subset=["RESPONSE"])
    values_df["MEANING"] = values_df["MEANING"].apply(clean_label_text)

    variables_df["QUESTION"] = variables_df["LABEL"]

    eval_df["Case_Id"] = eval_df["CASE"].apply(normalize_case_id)
    eval_df = eval_df[eval_df["Case_Id"].notna()].copy()
    eval_df["Case_Id"] = eval_df["Case_Id"].astype(int)

    eval_df["Questionnaire_Id"] = eval_df["QUESTNNR"].apply(normalize_questnnr)

    # Platform
    platform_df = pd.DataFrame(
        [{"Platform_Title": "Website", "Platform_Id": 1}],
        columns=["Platform_Title", "Platform_Id"],
    )

    # User
    user_ids = sorted(eval_df["Case_Id"].unique().tolist())
    user_df = pd.DataFrame(
        {
            "User_Id": user_ids,
            "Username": [f"Username_{user_id}" for user_id in user_ids],
        }
    )

    # Questionnaire
    questionnaire_df = pd.DataFrame(
        [
            {
                "Questionnarie_Id": qid,
                "Questionnarie_Type": QUESTIONNAIRE_TYPE_BY_ID[qid],
                "Description": qname,
            }
            for qid, qname in QUESTIONNAIRE_NAME_BY_ID.items()
        ],
        columns=["Questionnarie_Id", "Questionnarie_Type", "Description"],
    )

    variables_df = variables_df.copy()
    variables_df["_Questionnaire_Id"] = variables_df["VAR"].astype(str).apply(questionnaire_for_var)
    question_vars = variables_df[variables_df["_Questionnaire_Id"].notna()].copy()

    question_body = question_vars["QUESTION"].where(
        question_vars["QUESTION"].notna() & (question_vars["QUESTION"].astype(str).str.strip() != ""),
        question_vars["LABEL"],
    )

    question_df = pd.DataFrame(
        {
            "Question_Id": question_vars["VAR"].astype(str),
            "Question_Body": question_body.astype(str),
            "Questionnarie_Id": question_vars["_Questionnaire_Id"],
        }
    ).drop_duplicates(subset=["Question_Id"], keep="first")

    # Answer options from codebook values.
    codebook_answers = values_df[values_df["VAR"].isin(question_df["Question_Id"])].copy()
    type_lookup = (
        question_vars[["VAR", "TYPE"]]
        .drop_duplicates(subset=["VAR"])
        .set_index("VAR")["TYPE"]
        .to_dict()
    )

    codebook_answers = complete_sus_likert_options(codebook_answers)
    codebook_answers["_Answer_Rank"] = codebook_answers.groupby("VAR").cumcount() + 1

    answer_ids: list[int] = []
    for var_name, rank in zip(
        codebook_answers["VAR"].astype(str),
        codebook_answers["_Answer_Rank"].astype(int),
        strict=False,
    ):
        family_code, question_code = parse_question_components(var_name)
        # Format: FQQQQQRR => family + question_code + answer_rank
        # Example M002_01 option 1 -> 10020101
        answer_id = family_code * 10_000_000 + question_code * 100 + rank
        answer_ids.append(answer_id)

    answer_df = pd.DataFrame(
        {
            "Answer_Id": answer_ids,
            "Response_Code": codebook_answers["RESPONSE"].apply(normalize_response_code),
            "Response_Label": codebook_answers["MEANING"],
            "Type": codebook_answers["VAR"].map(type_lookup).apply(answer_type_for_variable),
            "Question_Id": codebook_answers["VAR"],
        }
    )

    max_answer_rank_by_question = (
        codebook_answers.groupby("VAR")["_Answer_Rank"].max().astype(int).to_dict()
    )

    synthetic_answers: list[dict[str, object]] = []
    textual_answer_lookup: dict[str, int] = {}
    null_answer_lookup: dict[str, int] = {}
    for question in question_df.itertuples(index=False):
        qid = str(question.Question_Id)
        family_code, question_code = parse_question_components(qid)
        next_rank = int(max_answer_rank_by_question.get(qid, 0)) + 1
        variable_type = type_lookup.get(qid)

        if str(variable_type).strip().upper() == "TEXT":
            textual_answer_id = family_code * 10_000_000 + question_code * 100 + next_rank
            textual_answer_lookup[qid] = textual_answer_id
            synthetic_answers.append(
                {
                    "Answer_Id": textual_answer_id,
                    "Response_Code": "TEXT",
                    "Response_Label": "Textual response",
                    "Type": "Textual",
                    "Question_Id": qid,
                }
            )
            next_rank += 1

        null_answer_id = family_code * 10_000_000 + question_code * 100 + next_rank
        null_answer_lookup[qid] = null_answer_id
        synthetic_answers.append(
            {
                "Answer_Id": null_answer_id,
                "Response_Code": "null",
                "Response_Label": "No recorded response",
                "Type": "NullAnswer",
                "Question_Id": qid,
            }
        )

    if synthetic_answers:
        answer_df = pd.concat([answer_df, pd.DataFrame(synthetic_answers)], ignore_index=True)

    # Relationship instances: one answered question per visit/question pair.
    response_rows: list[dict[str, object]] = []
    
    # Create lookup: (VAR, RESPONSE_CODE) -> ANSWER_ID
    answer_lookup = (
        answer_df.set_index(["Question_Id", "Response_Code"])["Answer_Id"]
        .to_dict()
    )
    
    # Create lookup: (VAR, RESPONSE) -> MEANING
    code_meaning_lookup = (
        answer_df[answer_df["Type"] != "NullAnswer"]
        .set_index(["Question_Id", "Response_Code"])["Response_Label"]
        .to_dict()
    )

    question_ids_by_questionnaire = {
        questionnaire_id: questions["Question_Id"].astype(str).tolist()
        for questionnaire_id, questions in question_df.groupby("Questionnarie_Id", sort=False)
    }
    for row in eval_df.itertuples(index=False):
        visit_id = int(getattr(row, "Case_Id"))
        user_id = int(getattr(row, "Case_Id"))
        questionnaire_id = getattr(row, "Questionnaire_Id")

        for qid in question_ids_by_questionnaire.get(questionnaire_id, []):
            raw_value = getattr(row, qid, pd.NA)
            if pd.isna(raw_value):
                response_str = NO_RECORDED_RESPONSE_VALUE
                answer_id = null_answer_lookup[qid]
            else:
                response_str = str(raw_value).strip()
                if not response_str:
                    response_str = NO_RECORDED_RESPONSE_VALUE
                    answer_id = null_answer_lookup[qid]
                else:
                    # First try: raw_value is already a code
                    response_code = normalize_response_code(response_str)
                    response_label = code_meaning_lookup.get((qid, response_code), pd.NA)

                    if pd.isna(response_label):
                        answer_id = textual_answer_lookup.get(qid, null_answer_lookup[qid])
                    else:
                        answer_id = answer_lookup.get((qid, response_code), null_answer_lookup[qid])

            response_rows.append(
                {
                    "Visit_Id": visit_id,
                    "User_Id": user_id,
                    "Platform_Id": 1,
                    "Questionnarie_Id": questionnaire_id,
                    "Question_Id": qid,
                    "Answer_Id": answer_id,
                    # Strip a trailing ".0" from whole-number raw values
                    # (e.g. "7.0" -> "7") without touching free text.
                    "Raw_Response_Value": normalize_response_code(response_str),
                }
            )

    response_df = pd.DataFrame(response_rows).sort_values(
        ["Visit_Id", "Question_Id"], kind="stable"
    )

    duplicate_response_rows = response_df.duplicated(
        subset=["Visit_Id", "Question_Id"],
        keep=False,
    )
    if duplicate_response_rows.any():
        duplicates = response_df.loc[
            duplicate_response_rows,
            ["Visit_Id", "Question_Id"],
        ].drop_duplicates()
        raise ValueError(
            "Response.csv requires one answer per Visit_Id + Question_Id. "
            f"Found duplicates: {duplicates.to_dict(orient='records')}"
        )

    # Export tables mapped to the CSV header names used by the source model.
    platform_export_df = platform_df.rename(
        columns={
            "Platform_Title": "PlatformTitle",
            "Platform_Id": "PlatformId",
        }
    )
    user_export_df = user_df.rename(
        columns={
            "User_Id": "UserId",
        }
    )
    questionnaire_export_df = questionnaire_df.rename(
        columns={
            "Questionnarie_Id": "QuestionnarieId",
            "Questionnarie_Type": "QuestionnarieType",
        }
    )
    question_export_df = question_df[
        ["Question_Id", "Question_Body", "Questionnarie_Id"]
    ].rename(
        columns={
            "Question_Id": "QuestionId",
            "Question_Body": "QuestionBody",
            "Questionnarie_Id": "QuestionnarieId",
        }
    )
    answer_export_df = answer_df[
        [
            "Answer_Id",
            "Question_Id",
            "Response_Code",
            "Response_Label",
            "Type",
        ]
    ].rename(
        columns={
            "Answer_Id": "AnswerId",
            "Question_Id": "QuestionId",
            "Response_Code": "ResponseCode",
            "Response_Label": "ResponseLabel",
        }
    )
    response_export_df = response_df.rename(
        columns={
            "Visit_Id": "VisitId",
            "User_Id": "UserId",
            "Platform_Id": "PlatformId",
            "Questionnarie_Id": "QuestionnarieId",
            "Question_Id": "QuestionId",
            "Answer_Id": "AnswerId",
            "Raw_Response_Value": "RawResponseValue",
        }
    )

    return {
        "platform": platform_export_df,
        "user": user_export_df,
        "questionnaire": questionnaire_export_df,
        "question": question_export_df,
        "answer": answer_export_df,
        "response": response_export_df,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tables = build_tables()

    tables["platform"].to_csv(OUTPUT_DIR / "Platform.csv", index=False, encoding="utf-8")
    tables["user"].to_csv(OUTPUT_DIR / "User.csv", index=False, encoding="utf-8")
    tables["question"].to_csv(OUTPUT_DIR / "Questions.csv", index=False, encoding="utf-8")
    tables["answer"].to_csv(OUTPUT_DIR / "Answer.csv", index=False, encoding="utf-8")
    tables["questionnaire"].to_csv(OUTPUT_DIR / "Questionnarie.csv", index=False, encoding="utf-8")
    tables["response"].to_csv(OUTPUT_DIR / "Response.csv", index=False, encoding="utf-8")

    for obsolete_file_name in ("Visit.csv", "Answered.csv", "RespondedTo.csv"):
        obsolete_file = OUTPUT_DIR / obsolete_file_name
        if obsolete_file.exists():
            obsolete_file.unlink()

    print("Export complete.")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Files:")
    for file_name in sorted(p.name for p in OUTPUT_DIR.glob("*.csv")):
        print(f"- {file_name}")


if __name__ == "__main__":
    main()

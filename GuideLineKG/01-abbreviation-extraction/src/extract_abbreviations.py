"""Extract abbreviation definitions from guideline markdown files.

Looks for three places guideline documents define abbreviations, in order
of trust (a later method never overrides an abbreviation already found by
an earlier one):

1. Tables with a column header containing "abbreviation" - either
   pipe-style markdown (| Abbreviation | Definition |) or raw <table>/<th>/
   <td> HTML embedded in the markdown (how PDF-to-markdown conversion
   usually renders tables) - most reliable, structured data.
2. Footnote/glossary paragraphs of the form
   "ACEI, angiotensin-converting enzyme inhibitor; ARB, angiotensin
   receptor blocker; VA, ventricular arrhythmia." (comma, colon, or
   "indicates" between the abbreviation and its meaning) - also explicit,
   just free text instead of a table.
3. Inline parenthetical definitions in sentences, either order:
   "ventricular arrhythmia (VA)" or "VA (ventricular arrhythmia)" -
   heuristic, lowest trust.

Each abbreviation is kept once (first definition found, by the priority
above). A plural form ("VAs") is folded onto its singular base ("VA")
rather than becoming a separate entry; the plural spelling is recorded in
plural_forms_seen so occurrences of it aren't mistaken for a new,
undefined abbreviation later.

Usage:
    python extract_abbreviations.py file1.md dir/with/mds -o abbreviations.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

ABBR_TOKEN = r"[A-Z][A-Za-z0-9\-/]{0,9}"

TABLE_DEF_HEADERS = {"definition", "meaning", "full form", "term", "description", "expansion"}

FOOTNOTE_SEGMENT = re.compile(rf"^\s*({ABBR_TOKEN})\s*(?:[,:]\s*|\bindicates\b\s*)(.+)$")

LONG_THEN_SHORT = re.compile(
    r"([A-Za-z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*){0,7})\s*\(([A-Z]{2,6}s?)\)"
)
SHORT_THEN_LONG = re.compile(
    r"\b([A-Z]{2,6})\b\s*\(([A-Za-z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*){0,7})\)"
)


def _clean_cell(cell: str) -> str:
    return re.sub(r"[*_`]", "", cell).strip()


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [_clean_cell(c) for c in line.split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells if c)


def _pick_columns(header: list[str]) -> tuple[int | None, int | None]:
    abbr_col = next((idx for idx, h in enumerate(header) if "abbrev" in h.lower()), None)
    if abbr_col is None:
        return None, None
    def_col = next((idx for idx, h in enumerate(header) if h.lower() in TABLE_DEF_HEADERS), None)
    if def_col is None:
        other_cols = [idx for idx in range(len(header)) if idx != abbr_col]
        def_col = other_cols[0] if other_cols else None
    return abbr_col, def_col


def extract_from_markdown_tables(text: str) -> list[tuple[str, str, str]]:
    found = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if "|" not in lines[i]:
            i += 1
            continue
        block = []
        while i < len(lines) and "|" in lines[i]:
            block.append(lines[i])
            i += 1
        if len(block) < 2:
            continue
        rows = [_split_row(line) for line in block]
        abbr_col, def_col = _pick_columns(rows[0])
        if abbr_col is None or def_col is None:
            continue
        for row in rows[1:]:
            if _is_separator_row(row) or abbr_col >= len(row) or def_col >= len(row):
                continue
            abbr, definition = row[abbr_col].strip(), row[def_col].strip()
            if abbr and definition and re.fullmatch(ABBR_TOKEN, abbr):
                found.append((abbr, definition, "table"))
    return found


def extract_from_html_tables(text: str) -> list[tuple[str, str, str]]:
    """Same idea as extract_from_markdown_tables, for raw <table> HTML -
    the format PDF-to-markdown conversion usually produces. One <table> tag
    can contain more than one logical table (a new all-<th> row starts a
    new header/column mapping), so the column mapping is re-picked every
    time a header row is seen rather than once per <table>."""
    found = []
    soup = BeautifulSoup(text, "html.parser")
    for table in soup.find_all("table"):
        abbr_col = def_col = None
        for row in table.find_all("tr"):
            header_cells = row.find_all("th")
            if header_cells:
                header = [c.get_text(" ", strip=True) for c in header_cells]
                abbr_col, def_col = _pick_columns(header)
                continue
            if abbr_col is None or def_col is None:
                continue
            cells = [c.get_text(" ", strip=True) for c in row.find_all("td")]
            if abbr_col >= len(cells) or def_col >= len(cells):
                continue
            abbr, definition = cells[abbr_col].strip(), cells[def_col].strip()
            if abbr and definition and re.fullmatch(ABBR_TOKEN, abbr):
                found.append((abbr, definition, "table"))
    return found


def _strip_html_tags(text: str) -> str:
    """Prose (figure captions, footnotes) can carry inline HTML like <u>...</u>
    or <sup>...</sup> - not just the <table> blocks handled separately - and
    a tag boundary landing mid-sentence would otherwise leak into an
    expansion (e.g. "tip pole of</u> electrode")."""
    return re.sub(r"<[^>]+>", " ", text)


def extract_from_footnotes(text: str) -> list[tuple[str, str, str]]:
    found = []
    text = _strip_html_tags(text)
    for para in re.split(r"\n\s*\n", text):
        para_flat = " ".join(para.split())
        segments = [s.strip() for s in para_flat.split(";") if s.strip()]
        matched = []
        for seg in segments:
            m = FOOTNOTE_SEGMENT.match(seg.rstrip("."))
            if m and m.group(2).strip():
                matched.append((m.group(1), m.group(2).strip(), "footnote"))
        if len(matched) >= 2:  # a single hit is too likely to be a false positive
            found.extend(matched)
    return found


def _sub_tokens(words: list[str]) -> list[tuple[str, int]]:
    """Flatten whitespace-split words into hyphen-split parts, each tagged
    with the index of the whitespace word it came from (so a compound like
    "cardioverter-defibrillator" contributes two initials, C and D)."""
    tokens = []
    for idx, word in enumerate(words):
        for part in word.split("-"):
            if part:
                tokens.append((part, idx))
    return tokens


def _match_backward(words_before: list[str], letters: str) -> str | None:
    """Shortest word span ending right before the "(" whose initials spell
    `letters`; returns None (reject) if the words there don't actually
    match, instead of guessing an over-long phrase."""
    tokens = _sub_tokens(words_before)
    if len(tokens) < len(letters):
        return None
    tail = tokens[-len(letters):]
    if "".join(t[0][0].upper() for t in tail) != letters.upper():
        return None
    return " ".join(words_before[tail[0][1]:])


def _match_forward(words_after: list[str], letters: str) -> str | None:
    tokens = _sub_tokens(words_after)
    if len(tokens) < len(letters):
        return None
    head = tokens[: len(letters)]
    if "".join(t[0][0].upper() for t in head) != letters.upper():
        return None
    return " ".join(words_after[: head[-1][1] + 1])


def extract_from_inline(text: str) -> list[tuple[str, str, str]]:
    found = []
    text = _strip_html_tags(text)
    for m in LONG_THEN_SHORT.finditer(text):
        abbr = m.group(2)
        phrase = _match_backward(m.group(1).split(), base_form(abbr))
        if phrase:
            found.append((abbr, phrase, "inline"))
    for m in SHORT_THEN_LONG.finditer(text):
        abbr = m.group(1)
        phrase = _match_forward(m.group(2).split(), base_form(abbr))
        if phrase:
            found.append((abbr, phrase, "inline"))
    return found


def base_form(abbr: str) -> str:
    if len(abbr) > 2 and abbr.endswith("s") and abbr[:-1].isupper():
        return abbr[:-1]
    return abbr


def collect_files(paths: list[str]) -> list[Path]:
    files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.md")))
        elif path.is_file():
            files.append(path)
    return files


def build_abbreviation_index(files: list[Path]) -> dict[str, dict]:
    results: dict[str, dict] = {}

    # Table and footnote hits are collected across ALL files before inline
    # hits, so a structured definition anywhere outranks a heuristic guess
    # from another file, regardless of file order.
    extractors = (
        extract_from_markdown_tables,
        extract_from_html_tables,
        extract_from_footnotes,
        extract_from_inline,
    )
    for extractor in extractors:
        for file in files:
            text = file.read_text(encoding="utf-8", errors="ignore")
            for abbr, definition, method in extractor(text):
                key = base_form(abbr)
                if key in results:
                    if abbr != key and abbr not in results[key]["plural_forms_seen"]:
                        results[key]["plural_forms_seen"].append(abbr)
                    continue
                results[key] = {
                    "abbreviation": key,
                    "expansion": definition,
                    "plural_forms_seen": [abbr] if abbr != key else [],
                    "source_file": str(file),
                    "method": method,
                }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Markdown files and/or directories to scan (directories are searched recursively for *.md)")
    parser.add_argument("-o", "--output", default="abbreviations.json", help="Output JSON path")
    args = parser.parse_args()

    files = collect_files(args.paths)
    if not files:
        raise SystemExit("No .md files found in the given paths")

    results = build_abbreviation_index(files)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Scanned {len(files)} file(s), found {len(results)} abbreviation(s) -> {out_path}")


if __name__ == "__main__":
    main()

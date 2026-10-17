"""Merge several extract_abbreviations.py output files into one.

Each per-document JSON already tags every abbreviation with the method that
found it (table/footnote/inline). Merging reuses that same trust order -
table beats footnote beats inline - across documents, so a clean table
definition in one guideline is kept over a heuristic inline guess in
another, and plural forms keep folding onto their singular base the same
way the extractor itself does.

Usage:
    python merge_abbreviations.py output_abb -o merged_abbreviations.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METHOD_PRIORITY = {"table": 0, "footnote": 1, "inline": 2}


def base_form(abbr: str) -> str:
    if len(abbr) > 2 and abbr.endswith("s") and abbr[:-1].isupper():
        return abbr[:-1]
    return abbr


def collect_json_files(paths: list[str]) -> list[Path]:
    files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
    return files


def merge(files: list[Path]) -> dict[str, dict]:
    # (entry, document name) pairs from every file, sorted by trust so the
    # dedup loop below always sees the most trustworthy definition first.
    entries = []
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        for entry in data.values():
            entries.append((entry, file.stem))
    entries.sort(key=lambda pair: METHOD_PRIORITY.get(pair[0]["method"], 99))

    results: dict[str, dict] = {}
    for entry, doc_name in entries:
        key = base_form(entry["abbreviation"])
        if key in results:
            for plural in entry.get("plural_forms_seen", []):
                if plural not in results[key]["plural_forms_seen"]:
                    results[key]["plural_forms_seen"].append(plural)
            if doc_name not in results[key]["source_documents"]:
                results[key]["source_documents"].append(doc_name)
            continue
        results[key] = {
            "abbreviation": key,
            "expansion": entry["expansion"],
            "plural_forms_seen": list(entry.get("plural_forms_seen", [])),
            "method": entry["method"],
            "source_documents": [doc_name],
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="Per-document abbreviation JSON files and/or directories containing them")
    parser.add_argument("-o", "--output", default="merged_abbreviations.json", help="Output JSON path")
    args = parser.parse_args()

    files = collect_json_files(args.inputs)
    if not files:
        raise SystemExit("No JSON files found in the given paths")

    total_entries = sum(len(json.loads(f.read_text(encoding="utf-8"))) for f in files)
    merged = merge(files)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Merged {len(files)} file(s), {total_entries} entries -> {len(merged)} unique abbreviation(s) -> {out_path}")


if __name__ == "__main__":
    main()

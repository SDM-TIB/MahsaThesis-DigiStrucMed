"""Convert an extract_abbreviations.py / merge_abbreviations.py output
(a JSON OBJECT keyed by abbreviation) into the array-of-objects shape
mapping-abbreviations.rml.ttl expects.

Why this step exists: everywhere else in this pipeline, RML iterates a
JSON ARRAY ("$.text_units[*]", "$[*]", ...) - a pattern already verified
against the installed RDFizer/jsonpath_ng. Iterating a JSON object's
values directly ("$.*") has no precedent here, so rather than rely on an
unverified JSONPath form, the object -> array conversion is done in
plain Python instead.

Usage:
    python stage_abbreviations.py path/to/merged_abbreviations.json -o abbreviations.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="extract_abbreviations.py/merge_abbreviations.py output JSON")
    parser.add_argument("-o", "--output", default="abbreviations.json", help="Staged array-of-objects JSON path")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = list(data.values())

    out_path = Path(args.output)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Staged {len(rows)} abbreviation(s) -> {out_path}")


if __name__ == "__main__":
    main()

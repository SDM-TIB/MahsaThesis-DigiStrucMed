"""Multi-file joint answer downsampling for global class fairness.

The BRINK pipeline's `scripts/auto_downsample_answers.sh` calls this file,
but it was missing from the repo — this is a faithful reconstruction from
the shell script's arguments and stated purpose.

What it does
------------
Some answer labels (e.g. a very common abbreviation method) dominate the QA
set, which biases training/eval. This step caps every answer so that no
single answer value exceeds `--max_ratio` of the *combined* row count across
ALL input files (train/val/test considered jointly — "global" fairness), by
randomly dropping the excess rows of over-represented answers.

Cap definition
--------------
    total        = number of rows across all input files
    cap_per_ans  = max(1, floor(max_ratio * total))
Any answer whose global count exceeds `cap_per_ans` is randomly downsampled
(seeded by --random_seed) to exactly `cap_per_ans` rows; answers at or below
the cap are kept in full. Dropped rows are removed from whichever split they
came from, so the splits stay disjoint and consistently balanced.

Safety
------
Originals are never modified. Each input `<name>.tsv` is written to
`<name><suffix>.tsv` (default suffix `_downsampled`). A before/after report
is printed so the result can be verified.

Usage (as the shell script calls it)
-------------------------------------
    python utils/auto_downsample_answers.py \
        --input train.tsv val.tsv test.tsv \
        --max_ratio 0.01 \
        --answer_column answer \
        --id_column id \
        --suffix _downsampled \
        --random_seed 42
"""
import argparse
import csv
import math
import os
import random
from collections import Counter, defaultdict

csv.field_size_limit(2 ** 31 - 1)


def read_tsv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return reader.fieldnames, list(reader)


def write_tsv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def output_path(input_path, suffix):
    root, ext = os.path.splitext(input_path)
    return f"{root}{suffix}{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="Jointly downsample answers across files so no answer "
                    "exceeds max_ratio of the global row count."
    )
    parser.add_argument("--input", nargs="+", required=True,
                        help="One or more TSV files (processed jointly).")
    parser.add_argument("--max_ratio", type=float, default=0.01,
                        help="Max fraction of global rows any single answer may hold.")
    parser.add_argument("--answer_column", default="answer")
    parser.add_argument("--id_column", default="id")
    parser.add_argument("--suffix", default="_downsampled")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.max_ratio <= 1:
        parser.error("--max_ratio must be in (0, 1].")

    rng = random.Random(args.random_seed)

    # ---- load every file, tagging each row with its source ----------------
    per_file = {}            # path -> (fieldnames, rows)
    answer_to_rows = defaultdict(list)  # answer_key -> [(path, row_index)]
    total = 0
    for path in args.input:
        fieldnames, rows = read_tsv(path)
        if args.answer_column not in (fieldnames or []):
            parser.error(f"{path}: no '{args.answer_column}' column (has {fieldnames}).")
        per_file[path] = (fieldnames, rows)
        for i, row in enumerate(rows):
            answer_to_rows[row[args.answer_column]].append((path, i))
            total += 1

    if total == 0:
        print("No rows found in input files; nothing to do.")
        return

    cap = max(1, math.floor(args.max_ratio * total))
    before = Counter({a: len(v) for a, v in answer_to_rows.items()})

    # ---- decide which rows to drop ----------------------------------------
    drop = set()  # (path, row_index)
    for answer, locations in answer_to_rows.items():
        if len(locations) > cap:
            keep = set(rng.sample(range(len(locations)), cap))
            for j, loc in enumerate(locations):
                if j not in keep:
                    drop.add(loc)

    # ---- write downsampled copies -----------------------------------------
    print("=" * 64)
    print(f"Global rows: {total:,} | answers: {len(before):,} | "
          f"max_ratio {args.max_ratio} -> cap {cap:,}/answer")
    over = [(a, c) for a, c in before.most_common() if c > cap]
    print(f"Answers over the cap (downsampled): {len(over)}")
    for a, c in over[:15]:
        print(f"   {a!r}: {c:,} -> {cap:,}")
    print("-" * 64)

    kept_total = 0
    for path, (fieldnames, rows) in per_file.items():
        kept = [row for i, row in enumerate(rows) if (path, i) not in drop]
        out = output_path(path, args.suffix)
        write_tsv(out, fieldnames, kept)
        kept_total += len(kept)
        print(f"{os.path.basename(path)}: {len(rows):,} -> {len(kept):,}  ->  {os.path.basename(out)}")

    print("-" * 64)
    print(f"Total: {total:,} -> {kept_total:,} rows kept "
          f"({total - kept_total:,} dropped). Originals untouched.")
    print("=" * 64)


if __name__ == "__main__":
    main()

"""Assign a unique, sequential global `rule_index` to every rule.

Referenced by scripts/complete_answer_search.sh (Step 2) but missing from the
repo — reconstructed here. grounding_generation.py names each rule's output
file after `meta["rule_index"]`; without it, every rule defaults to "unknown"
and the per-rule output files overwrite each other. This stamps a 0-based
index onto each rule in a `rules_string.json` (list of rule dicts) and writes
`rules_with_global_index.json`, matching the format used for qKG (same keys,
plus `rule_index`).

Usage:
    python utils/assign_global_rule_index.py \
        -i data/GuidelineKG/rules_string.json \
        -o data/GuidelineKG/rules_with_global_index.json
"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Add a sequential global rule_index to each rule."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input rules JSON (list of rule dicts with a 'rule' key).")
    parser.add_argument("-o", "--output", required=True,
                        help="Output path for the indexed rules JSON.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        rules = json.load(f)

    if not isinstance(rules, list):
        raise ValueError(f"{args.input}: expected a JSON list of rule objects.")

    for index, rule in enumerate(rules):
        rule["rule_index"] = index

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(rules)} rules with global index -> {args.output}")


if __name__ == "__main__":
    main()

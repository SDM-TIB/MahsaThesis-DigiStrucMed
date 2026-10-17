"""Run all four pipeline steps in order, one command instead of four
READMEs' worth of manual steps.

Order: step 1 (abbreviations) -> step 2 (Wikidata enrichment, needs step
1's output) -> step 3 (JSON knowledge graph) -> step 4 (merge). Steps 1 and
3 don't depend on each other but run in this fixed order anyway, for one
reason: it's simpler to reason about and re-run than "maybe parallel."

Stops immediately on the first failing step - never continues into a later
step against stale/partial output from a failed one. That's been the
actual source of nearly every broken-graph problem in this project so far
(see git history/session notes): a step half-failing, or being skipped,
while a later step silently uses whatever was already on disk from a
previous run.

Usage (from the repo root):
    py -3.12 run_pipeline.py
    py -3.12 run_pipeline.py --full          # data/level4-dataset-full instead
                                              # of the small working set -
                                              # much slower, hits Wikidata's
                                              # live SPARQL endpoint hundreds
                                              # of times for step 2
    py -3.12 run_pipeline.py --from 3        # skip steps 1-2, start at step 3
                                              # (e.g. after fixing something
                                              # in step 3 only)
    py -3.12 run_pipeline.py --from 2 --wikidata-from expand
                                              # step 2 only, skip straight to
                                              # expand_wikidata_entities.py -
                                              # e.g. after search_wikidata_-
                                              # candidates.py already finished
                                              # and wrote a good
                                              # wikidata_matches.json, but a
                                              # later sub-stage crashed. Without
                                              # this, --from 2 alone reruns ALL
                                              # of step 2 - including
                                              # search_wikidata_candidates.py's
                                              # own hundreds of live-SPARQL
                                              # batches - even though its output
                                              # was already sitting on disk,
                                              # untouched (it only writes at the
                                              # very end, on success).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], cwd: Path, extra_env: dict[str, str] | None = None) -> None:
    """subprocess.run with check=True - raises and stops the whole pipeline
    on a nonzero exit, rather than the "PowerShell keeps going after a
    failed line" trap every step's README warns about individually."""
    import os
    env = {**os.environ, **(extra_env or {})}
    print(f"  $ {' '.join(cmd)}   (cwd: {cwd.relative_to(ROOT)})", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def step_banner(number: int, title: str) -> None:
    print(f"\n{'=' * 70}\nStep {number}: {title}\n{'=' * 70}", flush=True)


def step1_abbreviations() -> None:
    step_banner(1, "Abbreviation extraction (01-abbreviation-extraction/)")
    base = ROOT / "01-abbreviation-extraction"
    per_document = base / "output" / "per_document"
    per_document.mkdir(parents=True, exist_ok=True)

    input_dirs = sorted(p for p in (base / "data" / "input_md").iterdir() if p.is_dir())
    if not input_dirs:
        raise SystemExit(f"No document folders found in {base / 'data' / 'input_md'}")
    for document_dir in input_dirs:
        run([sys.executable, "src/extract_abbreviations.py", str(document_dir),
             "-o", f"output/per_document/{document_dir.name}.json"], cwd=base)

    run([sys.executable, "src/merge_abbreviations.py", "output/per_document",
         "-o", "output/merged_abbreviations.json"], cwd=base)
    run([sys.executable, "src/stage_abbreviations.py", "output/merged_abbreviations.json",
         "-o", "output/abbreviations.json"], cwd=base)
    run([sys.executable, "-m", "rdfizer", "-c", "config/rdfizer-abbreviations.ini"],
        cwd=base, extra_env={"PYTHONUTF8": "1"})


WIKIDATA_SUBSTAGES = ["prepare", "search", "expand", "rdfizer"]


def step2_wikidata(dataset_dir: str, wikidata_from: str = "prepare") -> None:
    step_banner(2, "Wikidata enrichment (02-wikidata-enrichment/) - hits the live "
                   "Wikidata SPARQL endpoint, may take a while")
    base = ROOT / "02-wikidata-enrichment"
    abbreviations = "../01-abbreviation-extraction/output/merged_abbreviations.json"
    start = WIKIDATA_SUBSTAGES.index(wikidata_from)

    # Four sub-stages, each writing the file the next one reads - see this
    # step's own README for the full pipeline-stage writeup. Each is
    # individually skippable via --wikidata-from because they're NOT equally
    # cheap to redo: search_wikidata_candidates.py/expand_wikidata_entities.py
    # are hundreds of live-SPARQL batches each (checkpointed internally
    # against a mid-run crash - see their own docstrings - but neither
    # checkpoint survives a full process restart with fresh input, which is
    # exactly what plain `--from 2` causes). Skipping straight to a later
    # sub-stage only makes sense if its input file already exists on disk
    # from a previous run - that's on the caller, same as --from N itself
    # already requires earlier STEPS' output to exist.
    if start <= WIKIDATA_SUBSTAGES.index("prepare"):
        run([sys.executable, "src/prepare_wikidata_search.py", dataset_dir,
             "--abbreviations", abbreviations], cwd=base)
    if start <= WIKIDATA_SUBSTAGES.index("search"):
        run([sys.executable, "src/search_wikidata_candidates.py",
             "wikidata_search/wikidata_search_input.json"], cwd=base)
    if start <= WIKIDATA_SUBSTAGES.index("expand"):
        run([sys.executable, "src/expand_wikidata_entities.py",
             "wikidata_search/wikidata_matches.json"], cwd=base)
    if start <= WIKIDATA_SUBSTAGES.index("rdfizer"):
        run([sys.executable, "-m", "rdfizer", "-c", "config/rdfizer-wikidata-entities.ini"],
            cwd=base, extra_env={"PYTHONUTF8": "1"})


def step3_json_graph(dataset_dir: str) -> None:
    step_banner(3, "JSON knowledge graph (03-json-knowledge-graph/)")
    base = ROOT / "03-json-knowledge-graph"
    run([sys.executable, "src/run_json_batch.py", dataset_dir],
        cwd=base, extra_env={"PYTHONUTF8": "1"})


def step4_merge() -> None:
    step_banner(4, "Merge knowledge graphs (04-merge-knowledge-graphs/)")
    base = ROOT / "04-merge-knowledge-graphs"
    run([sys.executable, "src/merge_graphs.py"], cwd=base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true",
                         help="Use data/level4-dataset-full instead of the small working set "
                              "(539 files, slow, hits live Wikidata SPARQL hundreds of times)")
    parser.add_argument("--from", dest="from_step", type=int, choices=[1, 2, 3, 4], default=1,
                         help="Start at this step instead of 1 (e.g. --from 3 after fixing "
                              "only step 3; earlier steps' output must already exist)")
    parser.add_argument("--wikidata-from", choices=WIKIDATA_SUBSTAGES, default="prepare",
                         help="Within step 2, start at this sub-stage instead of from the top "
                              "(e.g. --wikidata-from expand after search_wikidata_candidates.py "
                              "already wrote a good wikidata_matches.json but a later sub-stage "
                              "crashed; that file must already exist). No effect unless step 2 "
                              "actually runs (i.e. --from is 1 or 2)")
    args = parser.parse_args()

    dataset_name = "data/level4-dataset-full" if args.full else "data/level4-dataset"
    step3_dataset = dataset_name
    step2_dataset = "../03-json-knowledge-graph/" + dataset_name

    steps = [
        (1, lambda: step1_abbreviations()),
        (2, lambda: step2_wikidata(step2_dataset, args.wikidata_from)),
        (3, lambda: step3_json_graph(step3_dataset)),
        (4, lambda: step4_merge()),
    ]

    start = time.monotonic()
    for number, step_fn in steps:
        if number < args.from_step:
            print(f"\nSkipping step {number} (--from {args.from_step})", flush=True)
            continue
        try:
            step_fn()
        except subprocess.CalledProcessError as exc:
            elapsed = time.monotonic() - start
            print(f"\n{'!' * 70}\nStep {number} FAILED (exit {exc.returncode}) after "
                  f"{elapsed:.0f}s - stopping. Fix the error above, then re-run with "
                  f"--from {number} to resume without redoing earlier steps.\n{'!' * 70}",
                  file=sys.stderr)
            raise SystemExit(1)

    elapsed = time.monotonic() - start
    output = ROOT / "04-merge-knowledge-graphs" / "output" / "knowledge_graph_merged.nt"
    print(f"\n{'=' * 70}\nDone in {elapsed:.0f}s -> {output.relative_to(ROOT)}\n{'=' * 70}")


if __name__ == "__main__":
    main()

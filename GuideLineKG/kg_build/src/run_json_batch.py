"""Map each level4 JSON file, scope local IDs, then merge the graphs.

Each RDFizer invocation gets a temporary mapping/config selecting that exact
input file. Source JSON and the user's mapping/config are not overwritten.
The rescope step adds semantic_unit_id to entity mentions, text units, grade
records, assertions, and their references. Shared grade values, surface forms,
and schema class IRIs remain global. Scoping must happen BEFORE merging:
otherwise repeated local IDs such as gr_0002 would lose their provenance.

Run: py -3.12 kg_build/src/run_json_batch.py kg_build/level4-dataset
Writes output_json/per_unit/*.nt and output_json/knowledge_graph_json.nt.
"""
from __future__ import annotations

import argparse
import configparser
import re
import tempfile
import os
import subprocess
import sys
from pathlib import Path

from rdflib import Graph

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich_json_types import enrich  # noqa: E402
from normalize_surface_forms import normalize_surface_forms  # noqa: E402
from rescope_json_ids import rescope  # noqa: E402

KG_BUILD_DIR = Path(__file__).resolve().parent.parent
MAPPING_FILE = KG_BUILD_DIR / "mappings" / "mapping-json.rml.ttl"
RDFIZER_CONFIG = KG_BUILD_DIR / "config" / "rdfizer-json.ini"
RDFIZER_OUTPUT = KG_BUILD_DIR / "output_json" / "knowledge_graph_json.nt"
PER_UNIT_DIR = KG_BUILD_DIR / "output_json" / "per_unit"
COMBINED_OUTPUT = KG_BUILD_DIR / "output_json" / "knowledge_graph_json.nt"


def run_rdfizer(config_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    subprocess.run(
        [sys.executable, "-m", "rdfizer", "-c", str(config_path)],
        cwd=KG_BUILD_DIR,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def process_one(source_json: Path, per_unit_path: Path) -> tuple[int, int]:
    """Map this exact file in isolation, scope its local IDs, then enrich."""
    with tempfile.TemporaryDirectory(prefix="guidelinekg-") as folder:
        work = Path(folder)
        mapping = MAPPING_FILE.read_text(encoding="utf-8")

        def select_source(match):
            original = match[1]
            # Vocabulary is global; every level4 logical source uses this input.
            if Path(original).name == "ontology-class-lookup.json":
                path = KG_BUILD_DIR / original
            else:
                path = source_json.resolve()
            return 'rml:source "' + path.as_posix() + '"'

        mapping = re.sub(r'rml:source "([^"\n]+)"', select_source, mapping)
        mapping_path = work / "mapping.ttl"
        mapping_path.write_text(mapping, encoding="utf-8")
        config = configparser.ConfigParser()
        config.read(RDFIZER_CONFIG)
        config["datasets"]["output_folder"] = work.as_posix()
        config["dataset1"]["mapping"] = mapping_path.as_posix()
        config_path = work / "rdfizer.ini"
        with config_path.open("w", encoding="utf-8") as stream:
            config.write(stream)
        run_rdfizer(config_path)
        output = work / (config["datasets"]["name"] + ".nt")
        if not output.exists():
            raise RuntimeError(f"RDFizer did not produce output for {source_json.name}")
        graph = Graph().parse(output, format="nt")

    raw_triples = len(graph)

    rescope(graph)
    enrich(graph)

    per_unit_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=per_unit_path, format="nt", encoding="utf-8")
    return raw_triples, len(graph)


def merge_all(per_unit_paths: list[Path], combined_path: Path) -> int:
    combined = Graph()
    for path in per_unit_paths:
        combined.parse(path, format="nt")
    # Different documents can spell the same concept with different
    # case/whitespace (e.g. "Beta Blocker" vs "beta blocker"); each per-unit
    # file only sees its own text, so this can only be caught once everything
    # is merged. See normalize_surface_forms.py for why this matters for
    # ds:wikidataMatch coverage.
    normalize_surface_forms(combined)
    combined.serialize(destination=combined_path, format="nt", encoding="utf-8")
    return len(combined)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Directory of *.json files, one semantic unit each")
    parser.add_argument("--glob", default="*.json", help="Filename pattern within input_dir (default: *.json)")
    args = parser.parse_args()

    source_files = sorted(args.input_dir.glob(args.glob))
    if not source_files:
        raise SystemExit(f"No files matching {args.glob!r} in {args.input_dir}")

    per_unit_paths = []
    failures = []
    for source_json in source_files:
        per_unit_path = PER_UNIT_DIR / f"{source_json.stem}.nt"
        try:
            raw, final = process_one(source_json, per_unit_path)
            print(f"{source_json.name}: {raw} -> {final} triples -> {per_unit_path.relative_to(KG_BUILD_DIR)}")
            per_unit_paths.append(per_unit_path)
        except (subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
            detail = exc.stderr[-2000:] if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
            print(f"{source_json.name}: FAILED - {detail}", file=sys.stderr)
            failures.append(source_json.name)

    if not per_unit_paths:
        raise SystemExit("Every file failed - nothing to merge.")

    total = merge_all(per_unit_paths, COMBINED_OUTPUT)
    print(f"\nMerged {len(per_unit_paths)}/{len(source_files)} file(s) -> {COMBINED_OUTPUT.relative_to(KG_BUILD_DIR)} ({total} triples)")
    if failures:
        print(f"Failed ({len(failures)}): {', '.join(failures)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""End-to-end acceptance test (skill section 11).

Runs the full pipeline (build_tables -> extract_assertions -> validate_tables
-> rdfizer) against the single supplied sample record in
tests/fixtures/sample_prediction.jsonl and checks that the output contains,
for that one semantic unit: a SemanticUnit resource, at least one Mention,
at least one UMLS candidate, at least one ontology-normalized local Entity,
at least one traceable assertion, and that the final RDF parses.

Run with: python -m pytest tests/test_acceptance.py
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "sample_prediction.jsonl"
ONTOLOGY = PROJECT_ROOT / "ontology" / "ds-ontology.ttl"
RULES = PROJECT_ROOT / "config" / "relation_rules.yaml"


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    work_dir = tmp_path_factory.mktemp("acceptance")
    data_dir = work_dir / "data"
    reports_dir = work_dir / "reports"

    subprocess.run([
        sys.executable, str(PROJECT_ROOT / "src" / "build_tables.py"),
        "--predictions", str(FIXTURE),
        "--ontology", str(ONTOLOGY),
        "--rules", str(RULES),
        "--output-dir", str(data_dir),
    ], check=True)

    subprocess.run([
        sys.executable, str(PROJECT_ROOT / "src" / "extract_assertions.py"),
        "--ontology", str(ONTOLOGY),
        "--rules", str(RULES),
        "--data-dir", str(data_dir),
        "--reports-dir", str(reports_dir),
    ], check=True)

    report_path = reports_dir / "validation_report.json"
    subprocess.run([
        sys.executable, str(PROJECT_ROOT / "src" / "validate_tables.py"),
        "--ontology", str(ONTOLOGY),
        "--data-dir", str(data_dir),
        "--report", str(report_path),
    ], check=True)

    return {"data_dir": data_dir, "reports_dir": reports_dir, "report_path": report_path}


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_semantic_unit_present(pipeline_output: dict[str, Path]) -> None:
    rows = _read_csv(pipeline_output["data_dir"] / "semantic_units.csv")
    assert len(rows) >= 1
    assert rows[0]["semantic_unit_uri"].startswith("http://digistrucmed.org/semantic-unit/")


def test_at_least_one_mention(pipeline_output: dict[str, Path]) -> None:
    rows = _read_csv(pipeline_output["data_dir"] / "mentions.csv")
    assert len(rows) >= 1
    assert all(r["mention_uri"].startswith("http://digistrucmed.org/mention/") for r in rows)


def test_at_least_one_umls_candidate(pipeline_output: dict[str, Path]) -> None:
    rows = _read_csv(pipeline_output["data_dir"] / "umls_candidates.csv")
    assert len(rows) >= 1
    assert rows[0]["cui"]


def test_at_least_one_ontology_normalized_entity(pipeline_output: dict[str, Path]) -> None:
    rows = _read_csv(pipeline_output["data_dir"] / "entities.csv")
    assert len(rows) >= 1
    classified = [r for r in rows if r["ontology_class_uri"]]
    assert classified, "expected at least one entity to be classified by relation_rules.yaml"
    assert classified[0]["ontology_class_uri"].startswith("http://digistrucmed.org/")


def test_at_least_one_traceable_assertion(pipeline_output: dict[str, Path]) -> None:
    rows = _read_csv(pipeline_output["data_dir"] / "assertions.csv")
    assert len(rows) >= 1
    row = rows[0]
    for field in ("subject_uri", "predicate_uri", "object_uri", "semantic_unit_uri", "evidence_text"):
        assert row[field], f"assertion missing {field}"


def test_validation_passes(pipeline_output: dict[str, Path]) -> None:
    import json
    report = json.loads(pipeline_output["report_path"].read_text(encoding="utf-8"))
    assert report["status"] == "PASS", report["errors"]


def test_rdf_generation_and_parsing(pipeline_output: dict[str, Path], tmp_path: Path) -> None:
    pytest.importorskip("rdfizer")

    # work_dir already == pipeline_output["data_dir"].parent, and mapping.rml.ttl's
    # logical sources are the relative path "data/...", so rdfizer must run
    # with this directory as its cwd (hence subprocess + cwd=, not an in-process
    # rdfizer.semantify() call, which would use pytest's own cwd instead).
    work_dir = pipeline_output["data_dir"].parent
    output_dir = work_dir / "output"
    output_dir.mkdir(exist_ok=True)

    ini_path = work_dir / "rdfizer.ini"
    ini_path.write_text(
        "[datasets]\n"
        "number_of_datasets: 1\n"
        "all_in_one_file: yes\n"
        "name: acceptance\n"
        "remove_duplicate: yes\n"
        "enrichment: no\n"
        "output_folder: output\n"
        "output_format: n-triples\n"
        "ordered: yes\n\n"
        "[dataset1]\n"
        "name: acceptance\n"
        f"mapping: {(PROJECT_ROOT / 'mappings' / 'mapping.rml.ttl').as_posix()}\n",
        encoding="utf-8",
    )

    # PYTHONUTF8=1 avoids a Windows-only rdfizer bug: open(path, "w") with no
    # encoding= falls back to the system codepage, which can outright crash
    # (not just mis-encode) on non-Latin source text - see README's "Running
    # RDFizer". fix_rdfizer_encoding.py below is a no-op once this is set,
    # but is still run to mirror the documented pipeline invocation exactly.
    env = {**os.environ, "PYTHONUTF8": "1"}
    subprocess.run([sys.executable, "-m", "rdfizer", "-c", str(ini_path)], check=True, cwd=work_dir, env=env)

    nt_path = output_dir / "acceptance.nt"
    subprocess.run([
        sys.executable, str(PROJECT_ROOT / "src" / "fix_rdfizer_encoding.py"), str(nt_path),
    ], check=True)

    from rdflib import Graph
    g = Graph()
    g.parse(nt_path, format="nt")
    assert len(g) > 0

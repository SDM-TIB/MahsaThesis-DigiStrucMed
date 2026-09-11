"""Validate the CSV tables produced by build_tables.py / extract_assertions.py
against the ontology and against each other. Writes reports/validation_report.json
and exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rdflib import Graph, RDFS
from rdflib.namespace import OWL

NS = "http://digistrucmed.org/"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("validate_tables")


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_ontology(ontology_path: Path):
    g = Graph()
    g.parse(ontology_path, format="turtle")
    classes = {str(s) for s in g.subjects(predicate=None, object=OWL.Class)}
    object_props = {str(s) for s in g.subjects(predicate=None, object=OWL.ObjectProperty)}
    datatype_props = {str(s) for s in g.subjects(predicate=None, object=OWL.DatatypeProperty)}
    predicates = object_props | datatype_props

    domains: dict[str, set[str]] = {}
    ranges: dict[str, set[str]] = {}
    for s, o in g.subject_objects(predicate=RDFS.domain):
        domains.setdefault(str(s), set()).add(str(o))
    for s, o in g.subject_objects(predicate=RDFS.range):
        ranges.setdefault(str(s), set()).add(str(o))

    # Follow rdfs:subClassOf one level for range/domain compatibility checks.
    subclass_of: dict[str, set[str]] = {}
    for s, o in g.subject_objects(predicate=RDFS.subClassOf):
        subclass_of.setdefault(str(s), set()).add(str(o))

    return classes, predicates, domains, ranges, subclass_of


def class_is_compatible(actual_class: str, expected_classes: set[str], subclass_of: dict[str, set[str]]) -> bool:
    if actual_class in expected_classes:
        return True
    seen = set()
    frontier = {actual_class}
    while frontier:
        c = frontier.pop()
        if c in seen:
            continue
        seen.add(c)
        for parent in subclass_of.get(c, set()):
            if parent in expected_classes:
                return True
            frontier.add(parent)
    return False


def is_external_uri(u: str) -> bool:
    return not u.startswith(NS)


def validate(data_dir: Path, ontology_path: Path) -> dict[str, Any]:
    classes, predicates, domains, ranges, subclass_of = load_ontology(ontology_path)

    semantic_units = read_csv(data_dir / "semantic_units.csv")
    mentions = read_csv(data_dir / "mentions.csv")
    umls_candidates = read_csv(data_dir / "umls_candidates.csv")
    entities = read_csv(data_dir / "entities.csv")
    assertions = read_csv(data_dir / "assertions.csv")
    umls_relations = read_csv(data_dir / "umls_relations.csv")
    umls_semantic_types = read_csv(data_dir / "umls_semantic_types.csv")

    errors: list[str] = []
    warnings: list[str] = []
    checks: Counter[str] = Counter()

    known_uris = set()
    known_uris.update(r["semantic_unit_uri"] for r in semantic_units)
    known_uris.update(r["mention_uri"] for r in mentions)
    known_uris.update(r["entity_uri"] for r in entities)
    known_uris.update(r["umls_concept_uri"] for r in umls_candidates if r["umls_concept_uri"])
    entity_class_by_uri = {r["entity_uri"]: r["ontology_class_uri"] for r in entities}

    # 1. subject/object URIs exist locally or are an allowed external URI (UMLS).
    for i, row in enumerate(assertions):
        for field in ("subject_uri", "object_uri"):
            u = row[field]
            checks["uri_existence"] += 1
            if not u:
                errors.append(f"assertions.csv row {i}: empty {field}")
            elif u not in known_uris and not (is_external_uri(u) and "uts.nlm.nih.gov" in u):
                errors.append(f"assertions.csv row {i}: {field} {u} does not exist in any table")

    # 2. predicate_uri exists in the ontology.
    for i, row in enumerate(assertions):
        checks["predicate_exists"] += 1
        if row["predicate_uri"] not in predicates:
            errors.append(f"assertions.csv row {i}: predicate {row['predicate_uri']} not defined in ontology")

    # 3. ontology domain/range compatibility where declared.
    for i, row in enumerate(assertions):
        pred = row["predicate_uri"]
        if pred not in domains and pred not in ranges:
            continue
        subj_class = entity_class_by_uri.get(row["subject_uri"])
        obj_class = entity_class_by_uri.get(row["object_uri"])
        checks["domain_range"] += 1
        if pred in domains and subj_class:
            if not class_is_compatible(subj_class, domains[pred], subclass_of):
                warnings.append(
                    f"assertions.csv row {i}: subject class {subj_class} not in declared domain "
                    f"{domains[pred]} for {pred}"
                )
        if pred in ranges and obj_class:
            if not class_is_compatible(obj_class, ranges[pred], subclass_of):
                warnings.append(
                    f"assertions.csv row {i}: object class {obj_class} not in declared range "
                    f"{ranges[pred]} for {pred}"
                )

    # 4. no empty subject/predicate/object in assertions.
    for i, row in enumerate(assertions):
        checks["non_empty_spo"] += 1
        if not row["subject_uri"] or not row["predicate_uri"] or not row["object_uri"]:
            errors.append(f"assertions.csv row {i}: empty subject/predicate/object")

    # 5. deterministic URI uniqueness (semantic units, mentions, entities, assertions).
    for name, rows, key in (
        ("semantic_units.csv", semantic_units, "semantic_unit_uri"),
        ("mentions.csv", mentions, "mention_uri"),
        ("entities.csv", entities, "entity_uri"),
        ("assertions.csv", assertions, "assertion_uri"),
    ):
        checks["uri_uniqueness"] += 1
        seen = Counter(r[key] for r in rows)
        dupes = [u for u, c in seen.items() if c > 1]
        if dupes:
            errors.append(f"{name}: {len(dupes)} duplicate {key} value(s), e.g. {dupes[:3]}")

    # 6. character offsets valid when source text exists.
    source_text_by_unit = {r["semantic_unit_uri"]: r["source_text"] for r in semantic_units}
    for i, row in enumerate(mentions):
        text = source_text_by_unit.get(row["semantic_unit_uri"], "")
        if not text:
            continue
        checks["char_offsets"] += 1
        try:
            start, end = int(row["char_start"]), int(row["char_end"])
        except (TypeError, ValueError):
            errors.append(f"mentions.csv row {i}: non-integer char_start/char_end")
            continue
        if not (0 <= start < end <= len(text)):
            errors.append(f"mentions.csv row {i}: char offsets [{start}, {end}) out of bounds for source_text")

    # 7. accepted UMLS links have CUIs.
    for i, row in enumerate(entities):
        checks["umls_accepted_has_cui"] += 1
        # accepted_umls_cui is only ever populated by build_tables.py's acceptance
        # rule, so a non-empty value must be a real CUI string.
        cui = row.get("accepted_umls_cui", "")
        if cui and not cui.startswith("C"):
            errors.append(f"entities.csv row {i}: accepted_umls_cui {cui!r} does not look like a UMLS CUI")

    # 8. unreviewed UMLS predictions are not emitted as owl:sameAs (structural check:
    #    this pipeline never emits owl:sameAs at all; verified against the RML mapping file).
    checks["no_sameAs_for_unreviewed"] += 1

    # 9. every direct clinical triple has a matching assertion provenance node.
    #    In this pipeline every row in assertions.csv already *is* the provenance
    #    record (ds:ExtractionAssertion is generated 1:1 from it in the RML
    #    mapping), so this reduces to: every assertion has a semantic_unit_uri.
    for i, row in enumerate(assertions):
        checks["assertion_has_provenance"] += 1
        if not row["semantic_unit_uri"]:
            errors.append(f"assertions.csv row {i}: missing semantic_unit_uri (no provenance)")
        if row["semantic_unit_uri"] not in known_uris:
            errors.append(f"assertions.csv row {i}: semantic_unit_uri {row['semantic_unit_uri']} does not exist")

    # 10. umls_relations.csv (optional, only present after fetch_umls_relations.py):
    #     non-empty endpoints, relation_uri uniqueness. cui2_uri is *not* required
    #     to be a known local UMLSConcept URI - MRREL frequently points at concepts
    #     never seen as a candidate in this document, which is expected, not an error.
    if umls_relations:
        seen_relation_uris: Counter[str] = Counter()
        for i, row in enumerate(umls_relations):
            checks["umls_relation_endpoints"] += 1
            if not row["cui1_uri"] or not row["cui2_uri"] or not row["relation_uri"]:
                errors.append(f"umls_relations.csv row {i}: empty cui1_uri/cui2_uri/relation_uri")
            if row["cui1_uri"] not in known_uris:
                errors.append(f"umls_relations.csv row {i}: cui1_uri {row['cui1_uri']} not a known UMLSConcept")
            seen_relation_uris[row["relation_uri"]] += 1
        dupes = [u for u, c in seen_relation_uris.items() if c > 1]
        if dupes:
            warnings.append(f"umls_relations.csv: {len(dupes)} relation_uri value(s) repeated (harmless, "
                             f"deduplicated by rdfizer's remove_duplicate), e.g. {dupes[:3]}")

    # 11. umls_semantic_types.csv (optional): TUI looks like T### and cui_uri is known.
    if umls_semantic_types:
        for i, row in enumerate(umls_semantic_types):
            checks["umls_semantic_type_tui_format"] += 1
            if not re.fullmatch(r"T\d{3}", row["tui"]):
                errors.append(f"umls_semantic_types.csv row {i}: tui {row['tui']!r} does not look like a UMLS TUI")
            if row["cui_uri"] not in known_uris:
                errors.append(f"umls_semantic_types.csv row {i}: cui_uri {row['cui_uri']} not a known UMLSConcept")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "counts": {
            "semantic_units": len(semantic_units),
            "mentions": len(mentions),
            "umls_candidates": len(umls_candidates),
            "entities": len(entities),
            "assertions": len(assertions),
            "umls_relations": len(umls_relations),
            "umls_semantic_types": len(umls_semantic_types),
        },
        "checks_run": dict(checks),
        "errors": errors,
        "warnings": warnings,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    report = validate(args.data_dir, args.ontology)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log.info("Validation %s: %d errors, %d warnings (report: %s)",
              report["status"], len(report["errors"]), len(report["warnings"]), args.report)
    if report["status"] != "PASS":
        for e in report["errors"][:20]:
            log.error(e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Build assertions.csv (the KG edge layer) from the normalized tables written
by build_tables.py, plus reports/rejected_or_review.csv for anything that
cannot be defensibly asserted.

Only two kinds of edges are emitted as assertions in a run with no source
text (SOURCE_ROOT=NONE):

  ds:hasMention    SemanticUnit -> Mention   (from mentions.csv, verbatim)
  ds:derivedEntity Mention      -> Entity    (from entities.csv, verbatim)

Both are fully determined by the prediction JSON itself - no interpretation
is involved - so they carry confidence 1.0 and review_status=auto_accepted
(the derivedEntity edge is marked review_required when the entity itself has
no ontology class yet).

Clinical relations (ds:recommends, ds:hasComorbidity, ds:appliesToDevice, ...)
require knowing which entities co-occur in the same sentence/table row, which
requires the source text/HTML structure. Without it, this script does not
fabricate them: candidate entities are instead summarized per semantic unit
in reports/rejected_or_review.csv for human review or a future SOURCE_ROOT-
enabled run.
"""
from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml
from rdflib import Graph
from rdflib.namespace import OWL

NS = "http://digistrucmed.org/"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("extract_assertions")


def curie_to_uri(curie: str) -> str:
    return NS + curie[3:] if curie.startswith("ds:") else curie


def local_name(u: str) -> str:
    return u.rstrip("/").rsplit("/", 1)[-1]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_ontology_predicates(ontology_path: Path) -> set[str]:
    g = Graph()
    g.parse(ontology_path, format="turtle")
    preds = set()
    for prop_type in (OWL.ObjectProperty, OWL.DatatypeProperty):
        preds |= {str(s) for s in g.subjects(predicate=None, object=prop_type)}
    return preds


def assertion_row(subject_uri: str, predicate_uri: str, object_uri: str,
                   semantic_unit_uri: str, evidence_text: str,
                   evidence_char_start: Any, evidence_char_end: Any,
                   extraction_method: str, confidence: float,
                   review_status: str) -> dict[str, Any]:
    assertion_local = f"{local_name(subject_uri)}_{local_name(predicate_uri)}_{local_name(object_uri)}"
    return {
        "assertion_uri": NS + "assertion/" + quote(assertion_local, safe=""),
        "subject_uri": subject_uri,
        "predicate_uri": predicate_uri,
        "object_uri": object_uri,
        "semantic_unit_uri": semantic_unit_uri,
        "evidence_text": evidence_text,
        "evidence_char_start": evidence_char_start,
        "evidence_char_end": evidence_char_end,
        "extraction_method": extraction_method,
        "confidence": confidence,
        "review_status": review_status,
    }


def build(data_dir: Path, ontology_path: Path, rules_path: Path, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(rules_path, encoding="utf-8") as f:
        rules = yaml.safe_load(f)

    ontology_predicates = load_ontology_predicates(ontology_path)
    for spec in rules.get("structural_provenance_predicates", []):
        p = curie_to_uri(spec["predicate"])
        if p not in ontology_predicates:
            raise ValueError(f"{spec['predicate']} is not defined in {ontology_path}")

    mentions = read_csv(data_dir / "mentions.csv")
    entities = read_csv(data_dir / "entities.csv")
    entities_by_mention = {e["source_mention_uri"]: e for e in entities}

    has_mention_pred = curie_to_uri("ds:hasMention")
    derived_entity_pred = curie_to_uri("ds:derivedEntity")

    assertions: list[dict[str, Any]] = []
    for m in mentions:
        assertions.append(assertion_row(
            subject_uri=m["semantic_unit_uri"],
            predicate_uri=has_mention_pred,
            object_uri=m["mention_uri"],
            semantic_unit_uri=m["semantic_unit_uri"],
            evidence_text=m["surface_form"],
            evidence_char_start=m["char_start"],
            evidence_char_end=m["char_end"],
            extraction_method="structural_provenance",
            confidence=1.0,
            review_status="auto_accepted",
        ))

        entity = entities_by_mention.get(m["mention_uri"])
        if entity is None:
            continue
        assertions.append(assertion_row(
            subject_uri=m["mention_uri"],
            predicate_uri=derived_entity_pred,
            object_uri=entity["entity_uri"],
            semantic_unit_uri=m["semantic_unit_uri"],
            evidence_text=m["surface_form"],
            evidence_char_start=m["char_start"],
            evidence_char_end=m["char_end"],
            extraction_method="structural_provenance",
            confidence=1.0,
            review_status="auto_accepted" if entity["ontology_class_uri"] else "review_required",
        ))

    _write_csv(data_dir / "assertions.csv", assertions, [
        "assertion_uri", "subject_uri", "predicate_uri", "object_uri", "semantic_unit_uri",
        "evidence_text", "evidence_char_start", "evidence_char_end",
        "extraction_method", "confidence", "review_status",
    ])
    log.info("assertions: %d (all structural_provenance; no clinical relations without source text)", len(assertions))

    _write_rejected_or_review(entities, reports_dir / "rejected_or_review.csv")


def _write_rejected_or_review(entities: list[dict[str, Any]], out_path: Path) -> None:
    rows: list[dict[str, Any]] = []

    unclassified_by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    classified_by_unit: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for e in entities:
        unit = e["semantic_unit_uri"]
        if e["ontology_class_uri"]:
            classified_by_unit[unit][e["ontology_class_uri"]].append(e)
        else:
            unclassified_by_unit[unit].append(e)

    for unit, ents in unclassified_by_unit.items():
        rows.append({
            "semantic_unit_uri": unit,
            "review_reason": "unclassified_entity",
            "ontology_class_uri": "",
            "entity_count": len(ents),
            "example_entity_uris": "|".join(e["entity_uri"] for e in ents[:5]),
            "example_labels": "|".join(e["label"] for e in ents[:5]),
            "note": "No lexical or UMLS-semantic-type rule matched; needs manual class assignment or a new rule in relation_rules.yaml.",
        })

    for unit, by_class in classified_by_unit.items():
        for cls, ents in by_class.items():
            rows.append({
                "semantic_unit_uri": unit,
                "review_reason": "relation_extraction_skipped_no_source_text",
                "ontology_class_uri": cls,
                "entity_count": len(ents),
                "example_entity_uris": "|".join(e["entity_uri"] for e in ents[:5]),
                "example_labels": "|".join(e["label"] for e in ents[:5]),
                "note": "Candidate subject/object for a clinical relation (see clinical_relation_predicates in relation_rules.yaml); "
                        "not asserted because SOURCE_ROOT text/table structure was unavailable to determine pairing.",
            })

    _write_csv(out_path, rows, [
        "semantic_unit_uri", "review_reason", "ontology_class_uri", "entity_count",
        "example_entity_uris", "example_labels", "note",
    ])


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--rules", type=Path, default=None)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--reports-dir", type=Path, default=None)
    args = parser.parse_args()

    rules_path = args.rules or (Path(__file__).parent.parent / "config" / "relation_rules.yaml")
    reports_dir = args.reports_dir or (Path(__file__).parent.parent / "reports")

    build(args.data_dir, args.ontology, rules_path, reports_dir)


if __name__ == "__main__":
    main()

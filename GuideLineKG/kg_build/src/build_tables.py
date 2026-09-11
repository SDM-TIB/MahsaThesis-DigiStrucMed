"""Normalize SciSpacy/UMLS prediction records into the flat CSV tables that
back the DigiStrucMed guideline knowledge graph: semantic_units, mentions,
umls_candidates and entities. Relation/assertion extraction happens
separately in extract_assertions.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import quote

import yaml
from bs4 import BeautifulSoup
from rdflib import Graph, RDFS
from rdflib.namespace import OWL

NS = "http://digistrucmed.org/"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_tables")


def uri(local_type: str, local_id: str) -> str:
    return f"{NS}{local_type}/{quote(local_id, safe='')}"


def load_ontology_classes(ontology_path: Path) -> set[str]:
    g = Graph()
    g.parse(ontology_path, format="turtle")
    classes = {str(s) for s in g.subjects(predicate=None, object=OWL.Class)}
    log.info("Loaded %d owl:Class definitions from %s", len(classes), ontology_path)
    return classes


def curie_to_uri(curie: str) -> str:
    if curie.startswith("ds:"):
        return NS + curie[3:]
    return curie


@dataclass
class LexicalRule:
    name: str
    ds_class: str
    match_type: str
    case_sensitive: bool
    normalization_confidence: str
    patterns: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    requires_semantic_type: list[str] = field(default_factory=list)
    _compiled: list[re.Pattern] = field(default_factory=list, repr=False)
    _value_set: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self._compiled = [re.compile(p, flags) for p in self.patterns]
        if self.case_sensitive:
            self._value_set = set(self.values)
        else:
            self._value_set = {v.lower() for v in self.values}

    def matches(self, text: str) -> bool:
        if self.match_type == "regex":
            return any(p.search(text) for p in self._compiled)
        if self.match_type == "exact":
            candidate = text if self.case_sensitive else text.lower()
            return candidate in self._value_set
        return False


class RuleEngine:
    def __init__(self, rules_path: Path, valid_classes: set[str]):
        with open(rules_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self.lexical_rules: list[LexicalRule] = []
        for r in raw.get("lexical_rules", []):
            ds_class_uri = curie_to_uri(r["ds_class"])
            if ds_class_uri not in valid_classes:
                log.warning("Skipping lexical rule %s: %s not in ontology", r["name"], r["ds_class"])
                continue
            self.lexical_rules.append(LexicalRule(
                name=r["name"],
                ds_class=ds_class_uri,
                match_type=r["match_type"],
                case_sensitive=r.get("case_sensitive", False),
                normalization_confidence=r.get("normalization_confidence", "medium"),
                patterns=r.get("patterns", []),
                values=r.get("values", []),
                requires_semantic_type=r.get("requires_semantic_type", []),
            ))

        self.semantic_type_rules: dict[str, dict[str, str]] = {}
        for tcode, spec in raw.get("semantic_type_rules", {}).items():
            ds_class_uri = curie_to_uri(spec["ds_class"])
            if ds_class_uri not in valid_classes:
                log.warning("Skipping semantic-type rule %s: %s not in ontology", tcode, spec["ds_class"])
                continue
            self.semantic_type_rules[tcode] = {
                "ds_class": ds_class_uri,
                "normalization_confidence": spec.get("normalization_confidence", "medium"),
            }

        acc = raw.get("umls_acceptance", {})
        self.accepted_min_score_field = acc.get("accepted", {}).get("min_score")
        self.accepted_require_no_flags = acc.get("accepted", {}).get("require_no_review_flags", True)
        self.accepted_require_label_status = acc.get("accepted", {}).get("require_label_status")
        self.close_match_min_score = acc.get("close_match", {}).get("min_score", 0.85)

        log.info(
            "Loaded %d lexical rules and %d semantic-type rules from %s",
            len(self.lexical_rules), len(self.semantic_type_rules), rules_path,
        )

    def classify_entity(self, mention: dict[str, Any]) -> tuple[str, str, str]:
        """Returns (ontology_class_uri, normalization_method, normalization_confidence)."""
        surface = mention.get("surface_form", "") or ""
        accepted_types = accepted_candidate_types(mention)

        for rule in self.lexical_rules:
            if not rule.matches(surface):
                continue
            if rule.requires_semantic_type and not (accepted_types & set(rule.requires_semantic_type)):
                continue
            return rule.ds_class, f"lexical_rule:{rule.name}", rule.normalization_confidence

        for tcode in accepted_types:
            spec = self.semantic_type_rules.get(tcode)
            if spec:
                return spec["ds_class"], f"semantic_type:{tcode}", spec["normalization_confidence"]

        return "", "unmapped", "none"

    def umls_link_status(self, mention: dict[str, Any], linker_threshold: float) -> str:
        """Returns 'accepted', 'close_match', or 'candidate_only'."""
        score = mention.get("predicted_score")
        flags = mention.get("review_flags") or []
        label_status = mention.get("label_status")
        if score is None:
            return "candidate_only"

        threshold = linker_threshold if self.accepted_min_score_field == "threshold_field" else self.accepted_min_score_field
        is_accepted = (
            score >= threshold
            and (not flags or not self.accepted_require_no_flags)
            and (label_status == self.accepted_require_label_status or self.accepted_require_label_status is None)
        )
        if is_accepted:
            return "accepted"
        if score >= self.close_match_min_score:
            return "close_match"
        return "candidate_only"


def accepted_candidate_types(mention: dict[str, Any]) -> set[str]:
    cui = mention.get("predicted_cui")
    for cand in mention.get("candidates", []) or []:
        if cand.get("cui") == cui:
            return set(cand.get("types", []) or [])
    return set()


def read_predictions(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        yield from data
    else:
        yield data


def extract_source_text(source_root: Path | None, export_unit_path: str) -> str:
    if not source_root or not export_unit_path:
        return ""
    candidate = source_root / export_unit_path
    if not candidate.is_file():
        return ""
    with open(candidate, encoding="utf-8", errors="replace") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def mention_local_id(semantic_unit_id: str, char_start: int, char_end: int, seen: dict[str, int]) -> str:
    base = f"{semantic_unit_id}_{char_start}_{char_end}"
    if base not in seen:
        seen[base] = 0
        return base
    seen[base] += 1
    log.warning("Duplicate mention offsets %s; disambiguating as _dup%d", base, seen[base])
    return f"{base}_dup{seen[base]}"


def build(predictions_path: Path, source_root: Path | None, ontology_path: Path,
          rules_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_classes = load_ontology_classes(ontology_path)
    engine = RuleEngine(rules_path, valid_classes)

    semantic_units_rows: list[dict[str, Any]] = []
    mentions_rows: list[dict[str, Any]] = []
    candidates_rows: list[dict[str, Any]] = []
    entities_rows: list[dict[str, Any]] = []

    n_units = n_mentions = n_candidates = n_entities_classified = 0
    n_source_found = n_source_missing = 0
    n_umls_accepted = n_umls_close = n_umls_candidate_only = 0

    for rec in read_predictions(predictions_path):
        n_units += 1
        semantic_unit_id = rec["semantic_unit_id"]
        semantic_unit_uri = uri("semantic-unit", semantic_unit_id)
        export_unit_path = rec.get("export_unit_path", "")

        source_text = extract_source_text(source_root, export_unit_path)
        if source_text:
            n_source_found += 1
        else:
            n_source_missing += 1

        semantic_units_rows.append({
            "semantic_unit_uri": semantic_unit_uri,
            "semantic_unit_id": semantic_unit_id,
            "document_sha256": rec.get("document_sha256", ""),
            "text_sha256": rec.get("text_sha256", ""),
            "benchmark_category": rec.get("benchmark_category", ""),
            "export_unit_path": export_unit_path,
            "stage4_relpath": rec.get("stage4_relpath", ""),
            "model_name": rec.get("model_name", ""),
            "linker_name": rec.get("linker_name", ""),
            "umls_source": rec.get("umls_source", ""),
            "review_status": rec.get("review_status", ""),
            "source_text": source_text,
        })

        linker_threshold = float(rec.get("linker_threshold", 0.7))
        seen_offsets: dict[str, int] = {}

        for mention in rec.get("mentions", []):
            n_mentions += 1
            char_start = mention.get("char_start")
            char_end = mention.get("char_end")
            local_id = mention_local_id(semantic_unit_id, char_start, char_end, seen_offsets)
            mention_uri = uri("mention", local_id)

            mentions_rows.append({
                "mention_uri": mention_uri,
                "semantic_unit_uri": semantic_unit_uri,
                "surface_form": mention.get("surface_form", ""),
                "char_start": char_start,
                "char_end": char_end,
                "ner_label": mention.get("ner_label", ""),
                "predicted_cui": mention.get("predicted_cui", ""),
                "predicted_score": mention.get("predicted_score", ""),
                "predicted_canonical_name": mention.get("predicted_canonical_name", ""),
                "label_status": mention.get("label_status", ""),
                "gold_status": mention.get("gold_status", ""),
                "review_flags": "|".join(mention.get("review_flags", []) or []),
            })

            for rank, cand in enumerate(mention.get("candidates", []) or [], start=1):
                cui = cand.get("cui", "")
                candidates_rows.append({
                    "candidate_uri": uri("umls-candidate", f"{local_id}_{rank}"),
                    "mention_uri": mention_uri,
                    "cui": cui,
                    "score": cand.get("score", ""),
                    "canonical_name": cand.get("canonical_name", ""),
                    "types": "|".join(cand.get("types", []) or []),
                    "candidate_rank": rank,
                    "umls_concept_uri": uri("umls", cui) if cui else "",
                    "umls_browser_url": f"https://uts.nlm.nih.gov/uts/umls/concept/{cui}" if cui else "",
                })
                n_candidates += 1

            link_status = engine.umls_link_status(mention, linker_threshold)
            if link_status == "accepted":
                n_umls_accepted += 1
            elif link_status == "close_match":
                n_umls_close += 1
            else:
                n_umls_candidate_only += 1

            ontology_class_uri, normalization_method, normalization_confidence = engine.classify_entity(mention)
            if ontology_class_uri:
                n_entities_classified += 1
            predicted_cui = mention.get("predicted_cui", "")
            accepted_cui = predicted_cui if link_status == "accepted" else ""
            accepted_umls_concept_uri = uri("umls", predicted_cui) if link_status == "accepted" and predicted_cui else ""
            close_match_umls_concept_uri = uri("umls", predicted_cui) if link_status == "close_match" and predicted_cui else ""

            entities_rows.append({
                "entity_uri": uri("entity", local_id),
                "label": mention.get("surface_form", ""),
                "normalized_label": re.sub(r"\s+", " ", (mention.get("surface_form") or "").strip()),
                "ontology_class_uri": ontology_class_uri,
                "source_mention_uri": mention_uri,
                "semantic_unit_uri": semantic_unit_uri,
                "normalization_method": normalization_method,
                "normalization_confidence": normalization_confidence,
                "review_status": "auto_classified" if ontology_class_uri else "review_required",
                "accepted_umls_cui": accepted_cui,
                "umls_link_status": link_status,
                "accepted_umls_concept_uri": accepted_umls_concept_uri,
                "close_match_umls_concept_uri": close_match_umls_concept_uri,
            })

    _write_csv(output_dir / "semantic_units.csv", semantic_units_rows, [
        "semantic_unit_uri", "semantic_unit_id", "document_sha256", "text_sha256",
        "benchmark_category", "export_unit_path", "stage4_relpath", "model_name",
        "linker_name", "umls_source", "review_status", "source_text",
    ])
    _write_csv(output_dir / "mentions.csv", mentions_rows, [
        "mention_uri", "semantic_unit_uri", "surface_form", "char_start", "char_end",
        "ner_label", "predicted_cui", "predicted_score", "predicted_canonical_name",
        "label_status", "gold_status", "review_flags",
    ])
    _write_csv(output_dir / "umls_candidates.csv", candidates_rows, [
        "candidate_uri", "mention_uri", "cui", "score", "canonical_name", "types",
        "candidate_rank", "umls_concept_uri", "umls_browser_url",
    ])
    _write_csv(output_dir / "entities.csv", entities_rows, [
        "entity_uri", "label", "normalized_label", "ontology_class_uri",
        "source_mention_uri", "semantic_unit_uri", "normalization_method",
        "normalization_confidence", "review_status", "accepted_umls_cui",
        "umls_link_status", "accepted_umls_concept_uri", "close_match_umls_concept_uri",
    ])

    log.info("semantic_units: %d (source_text found=%d, missing=%d)", n_units, n_source_found, n_source_missing)
    log.info("mentions: %d", n_mentions)
    log.info("umls_candidates: %d", n_candidates)
    log.info(
        "entities: %d (ontology-classified=%d, unmapped=%d)",
        len(entities_rows), n_entities_classified, len(entities_rows) - n_entities_classified,
    )
    log.info(
        "umls links: accepted=%d close_match=%d candidate_only=%d",
        n_umls_accepted, n_umls_close, n_umls_candidate_only,
    )

    _ensure_umls_enrichment_stubs(output_dir)


def _ensure_umls_enrichment_stubs(output_dir: Path) -> None:
    # src/fetch_umls_relations.py is optional (needs UMLS_API_KEY) and writes
    # these two tables when run. mapping.rml.ttl references them
    # unconditionally, and rdfizer's semantify() raises FileNotFoundError
    # (not just "0 rows") if a mapped CSV source is entirely absent - so a
    # header-only stub must exist here whenever the real enrichment hasn't
    # been run yet. Never overwrite an existing file: that would discard
    # already-fetched UMLS relations on every build_tables.py rerun.
    stubs = {
        "umls_relations.csv": ["cui1", "cui1_uri", "rel", "rela", "cui2", "cui2_uri", "cui2_name", "sab", "relation_uri"],
        "umls_semantic_types.csv": ["cui", "cui_uri", "tui", "semantic_type_name"],
    }
    for filename, fieldnames in stubs.items():
        path = output_dir / filename
        if not path.exists():
            _write_csv(path, [], fieldnames)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=None,
                         help="Root directory for export_unit_path HTML files, or omit if unavailable")
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--rules", type=Path, default=None,
                         help="Path to relation_rules.yaml (default: <this file's dir>/../config/relation_rules.yaml)")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rules_path = args.rules or (Path(__file__).parent.parent / "config" / "relation_rules.yaml")
    source_root = args.source_root if args.source_root and args.source_root.is_dir() else None
    if args.source_root and source_root is None:
        log.warning("--source-root %s not found; proceeding without source text", args.source_root)

    build(args.predictions, source_root, args.ontology, rules_path, args.output_dir)


if __name__ == "__main__":
    main()

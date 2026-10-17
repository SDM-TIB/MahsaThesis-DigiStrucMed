"""Union the three per-step .nt outputs and canonicalize surface-form nodes
that are pure spelling/inflection variants of the same concept.

Loading the three steps' outputs together already merges any two nodes that
share *identical* `ds:entity/{text}` text (see the root README's "How the
three graphs connect") - that's the whole integration mechanism, and it
needs no code. It does NOT merge nodes whose text differs by more than case/
whitespace, e.g. "beta-blocker" vs "beta-blockers" vs "beta blockers", or
"ACEI" vs "ACEIs": each stays its own node, independently typed, with its
own hasSourceMention/label/relation triples - the exact inconsistency this
script closes.

Two folds, applied only to `ds:SurfaceForm` nodes under `ds:entity/{text}`
(the same scope `normalize_surface_forms.py` uses), in order:

1. Case + whitespace + hyphen<->space fold (unconditional). "Beta-Blocker",
   "beta  blocker" and "beta-blocker" are, without exception, the same
   written concept - there's no plausible pair of distinct real-world
   referents that differ *only* by casing/spacing/hyphenation. This is a
   strict superset of normalize_surface_forms.py's own case/whitespace-only
   key (that script still runs standalone inside step 3 for its own
   within-step merge; this repeats the equivalent fold here so this script
   works even if step 3 wasn't re-run since a document was added).

2. Plural fold, closed-world only: merge two nodes whose already-folded keys
   differ by exactly one trailing "s" (e.g. "beta blocker" / "beta blockers",
   "acei" / "aceis") *only when both keys are already present as nodes in
   this graph*. This deliberately never invents an unseen singular by
   stemming every node - it only resolves a duplicate that demonstrably
   already exists in the data, so it cannot introduce a false merge for a
   word that happens to end in "s" without an actual singular counterpart
   on the other end (e.g. a genuine node called "status" is left alone,
   since "statu" is never a node).

Explicitly NOT folded: modifier/qualifier phrases like "evidence-based beta
blockers" are a distinct, more specific concept (the qualifier changes
meaning - e.g. a recommendation-strength distinction), not a spelling
variant, so they are left as their own node rather than merged into the
base concept. Only exact hyphen/space/case/simple-plural variants merge.

Run from `04-merge-knowledge-graphs/`:
    py -3.12 src/merge_graphs.py
Reads the three default per-step outputs (override with --sources), writes
output/knowledge_graph_merged.nt + output/merge_report.json.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote, unquote

from rdflib import Graph, Namespace, RDF, URIRef

DS = Namespace("http://digistrucmed.org/")
ENTITY_PREFIX = str(DS) + "entity/"

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = [
    ROOT / "03-json-knowledge-graph" / "output_json" / "knowledge_graph_json.nt",
    ROOT / "01-abbreviation-extraction" / "output" / "knowledge_graph_abbreviations.nt",
    ROOT / "02-wikidata-enrichment" / "wikidata_search" / "knowledge_graph_wikidata_entities.nt",
]
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _case_whitespace_hyphen_key(text: str) -> str:
    folded = text.replace("-", " ").casefold()
    return " ".join(folded.split())


def _singular_of(key: str) -> str | None:
    if len(key) > 3 and key.endswith("s") and not key.endswith("ss"):
        return key[:-1]
    return None


def load_graph(sources: list[Path]) -> Graph:
    graph = Graph()
    for path in sources:
        if not path.exists():
            raise SystemExit(f"Missing source graph: {path} "
                              f"(run that step's pipeline first - see its README)")
        graph.parse(path, format="nt")
    return graph


def _entity_nodes_by_key(graph: Graph) -> dict[str, set[URIRef]]:
    groups: dict[str, set[URIRef]] = defaultdict(set)
    for node in graph.subjects(RDF.type, DS.SurfaceForm):
        if not isinstance(node, URIRef) or not str(node).startswith(ENTITY_PREFIX):
            continue
        text = unquote(str(node)[len(ENTITY_PREFIX):])
        groups[_case_whitespace_hyphen_key(text)].add(node)
    return groups


def canonicalize(graph: Graph) -> tuple[int, list[dict]]:
    """Merge case/whitespace/hyphen and plural variants onto one canonical
    node each. Returns (#nodes remapped, report rows)."""
    groups = _entity_nodes_by_key(graph)
    key_of: dict[URIRef, str] = {node: key for key, nodes in groups.items() for node in nodes}
    abbreviation_nodes = set(graph.subjects(RDF.type, DS.Abbreviation))
    abbreviation_keys = {key_of[n] for n in abbreviation_nodes if n in key_of}

    key_parent: dict[str, str] = {key: key for key in groups}

    def find(key: str) -> str:
        while key_parent[key] != key:
            key = key_parent[key]
        return key

    def union(key_a: str, key_b: str) -> None:
        root_a, root_b = find(key_a), find(key_b)
        if root_a != root_b:
            key_parent[root_b] = root_a

    # 1. Trust the abbreviation pipeline's own hasPluralForm/pluralFormOf
    # edges for abbreviation plurals - it distinguishes a real plural ("VA"
    # -> "VAs") from an abbreviation that merely *looks* like one ending in
    # "S" but means something else ("OSA" obstructive sleep apnoea vs "OSAS"
    # obstructive sleep apnoea *syndrome"; "RAA" right atrial appendage vs
    # "RAAS" renin-angiotensin-aldosterone *system* - unrelated concepts
    # that happen to share a prefix). A blind trailing-"s" strip would wrongly
    # collide exactly these pairs, so abbreviations only ever merge via this
    # explicit relation, never via the heuristic in step 2.
    for abbr, plural in graph.subject_objects(DS.hasPluralForm):
        if abbr in key_of and plural in key_of:
            union(key_of[abbr], key_of[plural])
    for plural, abbr in graph.subject_objects(DS.pluralFormOf):
        if abbr in key_of and plural in key_of:
            union(key_of[abbr], key_of[plural])

    # 2. Closed-world plural fold for everything else (plain extracted
    # clinical terms, Wikidata-sourced labels) - only when BOTH the plural
    # and singular spelling already exist as nodes, and neither key is an
    # abbreviation (those are handled exclusively by step 1 above).
    for key in list(groups):
        if key in abbreviation_keys:
            continue
        singular = _singular_of(key)
        if singular is not None and singular in groups and singular not in abbreviation_keys:
            union(singular, key)

    merged_groups: dict[str, set[URIRef]] = defaultdict(set)
    for key, nodes in groups.items():
        merged_groups[find(key)] |= nodes

    remap: dict[URIRef, URIRef] = {}
    report = []
    for canonical_key, nodes in merged_groups.items():
        if len(nodes) <= 1:
            continue
        canonical = URIRef(ENTITY_PREFIX + quote(canonical_key, safe=""))
        variant_texts = sorted(unquote(str(n)[len(ENTITY_PREFIX):]) for n in nodes)
        report.append({"canonical": str(canonical), "merged_surface_forms": variant_texts})
        for node in nodes:
            if node != canonical:
                remap[node] = canonical

    if not remap:
        return 0, report

    rewritten = Graph()
    for subject, predicate, obj in graph:
        new_subject = remap.get(subject, subject)
        new_object = remap.get(obj, obj) if isinstance(obj, URIRef) else obj
        rewritten.add((new_subject, predicate, new_object))

    graph.remove((None, None, None))
    for triple in rewritten:
        graph.add(triple)
    return len(remap), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=Path, nargs="+", default=DEFAULT_SOURCES,
                         help="'.nt' files to union (default: the three steps' standard outputs)")
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT_DIR / "knowledge_graph_merged.nt")
    parser.add_argument("--report", type=Path, default=OUTPUT_DIR / "merge_report.json")
    args = parser.parse_args()

    graph = load_graph(args.sources)
    triples_before = len(graph)
    merged_count, report_rows = canonicalize(graph)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=args.output, format="nt", encoding="utf-8")
    args.report.write_text(json.dumps(report_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Unioned {len(args.sources)} source graph(s): {triples_before} -> {len(graph)} triples")
    print(f"Canonicalized {merged_count} spelling/inflection-variant node(s) "
          f"into {len(report_rows)} concept(s) -> {args.output}")
    print(f"Merge report ({len(report_rows)} groups) -> {args.report}")


if __name__ == "__main__":
    main()

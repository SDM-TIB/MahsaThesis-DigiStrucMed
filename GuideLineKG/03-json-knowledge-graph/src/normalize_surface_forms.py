"""Merge ds:entity/{surface_form} nodes that only differ by case/whitespace.

mapping-json.rml.ttl mints http://digistrucmed.org/entity/{surface_form}
straight from the extracted text (case preserved, see the "IRI-valued entity
fields" note in kg_build/README.md). "beta blocker", "Beta Blocker", and
"beta  blocker" therefore mint three distinct SurfaceForm nodes even though
they are the same lexical concept, and enrich_wikidata.py's ds:wikidataMatch
link only ever lands on whichever exact-cased node it happened to query for -
the others stay unmatched and disconnected from Wikidata. That fragmentation
grows with every new guideline document, since each one is mapped in
isolation (see run_json_batch.py) and RML has no case-folding function in
this RDFizer build.

This rewrites every ds:SurfaceForm node's IRI to a canonical form keyed by
casefold() + collapsed whitespace of its decoded text - the same
normalization enrich_wikidata.search_wikidata_api already applies when
choosing what to search for, so the merged node is exactly what a Wikidata
match will land on. All triples about the old, variant-cased nodes (labels,
hasSourceMention, wikidataMatch, ...) move onto the one canonical node;
rdflib's set semantics drop any resulting duplicate triples for free.

Run on the merged, cross-document graph (run_json_batch.py does this in
merge_all(), after combining all per-unit files) - not per-file, since the
same concept in two different casings is more likely to show up across
documents than within one. Safe to run twice (idempotent: a node already at
its canonical IRI is left alone).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote, unquote

from rdflib import Graph, Namespace, RDF, URIRef

DS = Namespace("http://digistrucmed.org/")
ENTITY_PREFIX = str(DS) + "entity/"


def _normalized_key(text: str) -> str:
    return " ".join(text.casefold().split())


def normalize_surface_forms(graph: Graph) -> int:
    groups: dict[str, set[URIRef]] = {}
    for node in graph.subjects(RDF.type, DS.SurfaceForm):
        if not isinstance(node, URIRef) or not str(node).startswith(ENTITY_PREFIX):
            continue
        text = unquote(str(node)[len(ENTITY_PREFIX):])
        groups.setdefault(_normalized_key(text), set()).add(node)

    remap = {}
    for key, nodes in groups.items():
        canonical = URIRef(ENTITY_PREFIX + quote(key, safe=""))
        for node in nodes:
            if node != canonical:
                remap[node] = canonical
    if not remap:
        return 0

    rewritten = Graph()
    for subject, predicate, obj in graph:
        new_subject = remap.get(subject, subject)
        new_object = remap.get(obj, obj) if isinstance(obj, URIRef) else obj
        rewritten.add((new_subject, predicate, new_object))

    graph.remove((None, None, None))
    for triple in rewritten:
        graph.add(triple)
    return len(remap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="N-Triples graph to normalize")
    parser.add_argument("--output", type=Path, default=None, help="Write here instead of overwriting --path")
    args = parser.parse_args()

    graph = Graph()
    graph.parse(args.path, format="nt")

    merged = normalize_surface_forms(graph)

    out_path = args.output or args.path
    graph.serialize(destination=out_path, format="nt", encoding="utf-8")
    print(f"{args.path}: {merged} surface-form node(s) merged -> {out_path}")


if __name__ == "__main__":
    main()

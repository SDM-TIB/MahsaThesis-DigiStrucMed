"""Strip percent-encoding out of digistrucmed.org IRIs for display/training use.

Only IRIs templated directly from free-text (see mapping-json.rml.ttl) ever
carry percent-encoding: the `entity/{surface_form}` namespace, plus a
handful of grade-value individuals built from their text (e.g.
`ClassOfRecommendation_3%3A%20Harm`). Every other IRI in the graph
(entity-mention/, text-unit/, etc.) is already a clean hash- or id-based
identifier and has no '%' to touch, so scanning the whole domain is safe -
this only ever rewrites nodes that actually contain a percent-encoded
sequence.

Percent-decoding is lossless: `%20` -> space, `%28`/`%29` -> `(` `)`,
`%5B`/`%5D` -> `[` `]`, `%E2%80%93` -> the original UTF-8 character, etc. -
nothing here changes what text an IRI represents, only how it's spelled out.

The RDF 1.1 N-Triples grammar technically allows any Unicode character
unescaped in an IRIREF, but plenty of tools (IDE linters, some SPARQL
engines/rule miners) still assume the older, ASCII-only RFC 3986 URI
grammar and reject raw non-ASCII. So after decoding:
- whitespace/control characters become `_` (the one substitution - see
  below for why it can't collide with anything);
- ASCII punctuation that's legal in a plain URI (hyphens, commas, parens,
  brackets, colons, etc.) is left completely untouched;
- anything else - non-ASCII characters, and the handful of characters an
  IRIREF forbids outright (<, >, ", {, }, |, ^, backtick, backslash), plus
  any stray `%` not part of a valid escape - is percent-encoded, same as
  the original, so it round-trips losslessly and stays valid everywhere.

This means only symbol-/foreign-script-heavy entities (accented letters,
Greek letters, em/en dashes, non-Latin scripts, etc.) keep any `%XX` at
all; the vast majority of plain-English entity names end up fully
readable. Using `_` only for whitespace also means this never merges any
nodes that merge_graphs.py's own hyphen/space/case fold didn't already
merge: e.g. "beta_blocker" (was "beta blocker") and "beta-blocker" stay
distinct, since only whitespace becomes `_`.

Run from `04-merge-knowledge-graphs/`:
    py -3.12 src/clean_entity_iris.py
Reads output/knowledge_graph_merged.nt, writes
output/knowledge_graph_merged_clean.nt + output/clean_entity_iris_report.json
(the report lists any collisions - two distinct decoded strings that landed
on the same cleaned IRI - so they can be reviewed).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote, unquote

from rdflib import Graph, URIRef

DOMAIN_PREFIX = "http://digistrucmed.org/"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
DEFAULT_INPUT = OUTPUT_DIR / "knowledge_graph_merged.nt"

_WHITESPACE_OR_CONTROL = re.compile(r"[\s\x00-\x1f\x7f]+")

# RFC 3986 'pchar' (the characters a URI path *segment* may contain raw):
# unreserved / pct-encoded / sub-delims / ":" / "@". Notably this excludes
# "[", "]", "?", "#" - those are gen-delims, legal only as structural
# separators (e.g. IPv6 host literals, query/fragment markers), not as
# literal content, even though they're printable ASCII. Strict validators
# (RDF4J/GraphDB's SailException "Invalid IRI value" among them) enforce
# this, so anything outside this whitelist gets percent-encoded rather than
# left raw - safer than maintaining a blacklist of "characters seen so far
# that turned out to be illegal".
_SAFE_RAW_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "-._~!$&'()*+,;=:@/"
)


def encode_local_text(text: str) -> str:
    """Collapse whitespace to `_` and percent-encode anything outside the
    RFC 3986 pchar whitelist. Shared with humanize_mention_iris.py, which
    applies this to plain-text labels (no percent-decoding needed there)."""
    collapsed = _WHITESPACE_OR_CONTROL.sub("_", text).strip("_")
    return "".join(ch if ch in _SAFE_RAW_CHARS else quote(ch, safe="") for ch in collapsed)


def clean_local_name(local: str) -> str:
    decoded = unquote(local, encoding="utf-8", errors="replace")
    return encode_local_text(decoded)


def build_remap(graph: Graph) -> tuple[dict[URIRef, URIRef], list[dict]]:
    by_new_full: dict[str, set[str]] = defaultdict(set)
    old_to_new: dict[URIRef, URIRef] = {}

    candidate_nodes = {
        node
        for node in set(graph.subjects()) | set(graph.objects())
        if isinstance(node, URIRef)
        and str(node).startswith(DOMAIN_PREFIX)
        and "%" in str(node)[len(DOMAIN_PREFIX):]
    }
    for node in candidate_nodes:
        old_local = str(node)[len(DOMAIN_PREFIX):]
        new_local = clean_local_name(old_local)
        new_full = DOMAIN_PREFIX + new_local
        by_new_full[new_full].add(old_local)
        new_node = URIRef(new_full)
        if new_node != node:
            old_to_new[node] = new_node

    collisions = [
        {"new_iri": new_full, "original_locals": sorted(old_locals)}
        for new_full, old_locals in by_new_full.items()
        if len(old_locals) > 1
    ]
    return old_to_new, collisions


def apply_remap(graph: Graph, remap: dict[URIRef, URIRef]) -> Graph:
    rewritten = Graph()
    for subject, predicate, obj in graph:
        new_subject = remap.get(subject, subject)
        new_object = remap.get(obj, obj) if isinstance(obj, URIRef) else obj
        rewritten.add((new_subject, predicate, new_object))
    return rewritten


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT_DIR / "knowledge_graph_merged_clean.nt")
    parser.add_argument("--report", type=Path, default=OUTPUT_DIR / "clean_entity_iris_report.json")
    args = parser.parse_args()

    graph = Graph()
    graph.parse(args.input, format="nt")
    triples_before = len(graph)

    remap, collisions = build_remap(graph)
    cleaned = apply_remap(graph, remap) if remap else graph

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.serialize(destination=args.output, format="nt", encoding="utf-8")
    args.report.write_text(
        json.dumps({"iris_rewritten": len(remap), "collisions": collisions}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Parsed {triples_before} triples from {args.input}")
    print(f"Rewrote {len(remap)} percent-encoded IRI(s) -> {args.output}")
    print(f"Found {len(collisions)} collision group(s) (two distinct spellings landing on one IRI) -> {args.report}")


if __name__ == "__main__":
    main()

"""Give document-hash-scoped IRIs readable local names instead of
`lu_<64-hex-sha256>_n####/...`.

The hash is the document's own `source.document_sha256` (see
rescope_json_ids.py and mapping-json.rml.ttl:141) - opaque by design, since
it's what makes IDs globally unique across documents. But every document
already has a human-readable slug sitting in the graph: `entity/*` nodes
carry `ds:sourceDocument` literals shaped like "<slug>_<sha256[:12]>" (e.g.
"zeppenfeld-2022-esc-guidelines-va-and-scd_d1710bde778e"), one per source
document. Matching a node's hash[:12] against that suffix recovers the same
slug with no external manifest.

Two passes, both keyed off that same doc_slugs lookup:

1. `entity-mention/` specifically also gets its own `rdfs:label` (the
   surface text it points at - see mapping-json.rml.ttl:317-330, added for
   exactly this "unlabeled mention nodes are unreadable" reason) prepended:

       entity-mention/lu_d1710bde...4230f712_n0019/e_0199
       -> entity-mention/lqts__zeppenfeld-2022-esc-guidelines-va-and-scd_n0019/e_0199

2. Every other `lu_<hash>` occurrence anywhere in a digistrucmed.org IRI -
   text-unit/, normative-assertion/, grade-record/, semantic-unit/, and any
   child resource of those (e.g. text-unit/.../locator) - has no natural
   short label, so just the hash itself is swapped for the slug:

       normative-assertion/lu_66f14f76...358e7_n0067/na_0003
       -> normative-assertion/mcdonagh-2023-esc-guidelines_n0067/na_0003

Neither pass can introduce a collision: the `_n####/<xx_####>` suffix was
already globally unique on its own (that's the whole point of embedding the
document hash), and the hash->slug lookup is a fixed one-to-one mapping, so
substituting the hash for its slug (and, for mentions, prepending a label)
is purely cosmetic.

Run from `04-merge-knowledge-graphs/`, after clean_entity_iris.py:
    py -3.12 src/humanize_mention_iris.py
Reads output/knowledge_graph_merged_clean.nt (in place by default), writes
output/humanize_mention_iris_report.json (anything with no label and/or no
matching document slug - so it can be reviewed; should be empty).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rdflib import RDFS, Graph, URIRef

from clean_entity_iris import DOMAIN_PREFIX, encode_local_text

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
DEFAULT_INPUT = OUTPUT_DIR / "knowledge_graph_merged_clean.nt"

_SOURCE_DOC_LITERAL = re.compile(r"^([a-z0-9-]+)_([0-9a-f]{12})$")
_MENTION_IRI = re.compile(
    r"^entity-mention/lu_([0-9a-f]{64})_n(\d+)/e_(\d+)(/.*)?$"
)
_GENERIC_LU_HASH = re.compile(r"lu_([0-9a-f]{64})")


def build_doc_slug_map(graph: Graph) -> dict[str, str]:
    source_doc = URIRef(DOMAIN_PREFIX + "sourceDocument")
    slug_by_prefix: dict[str, str] = {}
    for _, _, literal in graph.triples((None, source_doc, None)):
        m = _SOURCE_DOC_LITERAL.match(str(literal))
        if m:
            slug_by_prefix[m.group(2)] = m.group(1)
    return slug_by_prefix


def build_mention_labels(graph: Graph) -> dict[tuple[str, str, str], str]:
    labels: dict[tuple[str, str, str], str] = {}
    for subject, _, literal in graph.triples((None, RDFS.label, None)):
        if not isinstance(subject, URIRef) or not str(subject).startswith(DOMAIN_PREFIX):
            continue
        m = _MENTION_IRI.match(str(subject)[len(DOMAIN_PREFIX):])
        if m and not m.group(4):  # base mention node only, not a /normalization child
            labels[(m.group(1), m.group(2), m.group(3))] = str(literal)
    return labels


def build_remap(
    graph: Graph, doc_slugs: dict[str, str], mention_labels: dict[tuple[str, str, str], str]
) -> tuple[dict[URIRef, URIRef], list[dict]]:
    remap: dict[URIRef, URIRef] = {}
    unresolved: list[dict] = []

    candidate_nodes = {
        node
        for node in set(graph.subjects()) | set(graph.objects())
        if isinstance(node, URIRef) and str(node).startswith(DOMAIN_PREFIX + "entity-mention/lu_")
    }
    for node in candidate_nodes:
        local = str(node)[len(DOMAIN_PREFIX):]
        m = _MENTION_IRI.match(local)
        if not m:
            continue
        doc_hash, n, e, suffix = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        key = (doc_hash, n, e)

        slug = doc_slugs.get(doc_hash[:12])
        label = mention_labels.get(key)
        if slug is None or label is None:
            unresolved.append({
                "iri": str(node), "missing_slug": slug is None, "missing_label": label is None,
            })
            continue

        new_local = f"entity-mention/{encode_local_text(label)}__{slug}_n{n}/e_{e}{suffix}"
        new_node = URIRef(DOMAIN_PREFIX + new_local)
        if new_node != node:
            remap[node] = new_node

    return remap, unresolved


def build_generic_lu_remap(
    graph: Graph, doc_slugs: dict[str, str]
) -> tuple[dict[URIRef, URIRef], list[dict]]:
    """Swap any remaining `lu_<hash>` substring - text-unit/,
    normative-assertion/, grade-record/, semantic-unit/, and their child
    resources - for its document slug. Run this after build_remap() so
    entity-mention/ nodes (already rewritten with a label prefix) are
    skipped automatically - they no longer contain `lu_`."""
    remap: dict[URIRef, URIRef] = {}
    unresolved: list[dict] = []

    candidate_nodes = {
        node
        for node in set(graph.subjects()) | set(graph.objects())
        if isinstance(node, URIRef) and "lu_" in str(node) and str(node).startswith(DOMAIN_PREFIX)
    }
    for node in candidate_nodes:
        original = str(node)

        def _sub(m: re.Match) -> str:
            doc_hash = m.group(1)
            slug = doc_slugs.get(doc_hash[:12])
            if slug is None:
                unresolved.append({"iri": original, "hash": doc_hash})
                return m.group(0)
            return slug

        new_iri = _GENERIC_LU_HASH.sub(_sub, original)
        if new_iri != original:
            remap[node] = URIRef(new_iri)

    return remap, unresolved


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
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=OUTPUT_DIR / "humanize_mention_iris_report.json")
    args = parser.parse_args()

    graph = Graph()
    graph.parse(args.input, format="nt")
    triples_before = len(graph)

    doc_slugs = build_doc_slug_map(graph)
    mention_labels = build_mention_labels(graph)
    mention_remap, mention_unresolved = build_remap(graph, doc_slugs, mention_labels)
    graph = apply_remap(graph, mention_remap) if mention_remap else graph

    generic_remap, generic_unresolved = build_generic_lu_remap(graph, doc_slugs)
    rewritten = apply_remap(graph, generic_remap) if generic_remap else graph

    unresolved = mention_unresolved + generic_unresolved
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rewritten.serialize(destination=args.output, format="nt", encoding="utf-8")
    args.report.write_text(
        json.dumps({"documents_found": doc_slugs, "unresolved": unresolved}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Parsed {triples_before} triples from {args.input}")
    print(f"Found {len(doc_slugs)} document slug(s): {sorted(doc_slugs.values())}")
    print(f"Rewrote {len(mention_remap)} entity-mention IRI(s) with a label prefix")
    print(f"Rewrote {len(generic_remap)} other lu_<hash>-scoped IRI(s) (text-unit/normative-assertion/"
          f"grade-record/semantic-unit) -> {args.output}")
    print(f"Unresolved (no label and/or no matching doc slug): {len(unresolved)} -> {args.report}")


if __name__ == "__main__":
    main()

"""Make mapping-json.rml.ttl's per-file-only-unique URIs safe to merge
across many source JSON files (e.g. everything in level4-dataset/).

mappings/mapping-json.rml.ttl's header comment ("JSON fan-out/scoping
rules") documents why ds:ExtractedEntity/ds:TextUnit/ds:NormativeAssertion/
ds:SourceGradeRecord URIs are built from entity_id/text_unit_id/
assertion_id/grade_record_id ALONE (e.g. http://digistrucmed.org/entity-mention/
e_0001): a nested JSONPath iterator (entities[*], normative_assertions[*],
source_grade_records[*]) cannot see its ancestor semantic_unit_id, so RML
itself cannot embed it in those URIs. Those short ids are only unique
*within* one source file - across level4-dataset/'s files they collide
(every file restarts at e_0001, tu_0001, na_0001, gr_0001, ...), which would
silently merge unrelated entities/assertions from different guidelines onto
one resource if their per-file .nt outputs were combined as-is.

Shared surface-form objects (entity/Heart, for example) are excluded.
Occurrence nodes now use entity-mention/; legacy entity/ IDs remain supported.

This script rewrites the occurrence URI families to
http://digistrucmed.org/{kind}/{semantic_unit_id}/{local_id}[/suffix],
scoped by the ds:semanticUnitId already present in the same graph (which
embeds the full document_sha256 and is already globally unique - see the
same header comment). ds:Guideline/ds:Page/ds:SemanticUnit URIs
(document_sha256/page_id/semantic_unit_id-keyed) are untouched: they were
already globally unique and don't need this.

Run once per file's RDFizer output, before merging - see
src/run_json_batch.py, which does this automatically for a whole directory.
Safe to run twice on the same file (idempotent: a URI already scoped with
its own semantic_unit_id is left alone).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rdflib import Graph, Namespace, URIRef

DS = Namespace("http://digistrucmed.org/")

# Shared entity/ surface forms and entity-type/ codes must remain global.
RESCOPED_PREFIXES = [
    "http://digistrucmed.org/entity/",  # Compatibility with older ID-based graphs.
    "http://digistrucmed.org/entity-mention/",
    "http://digistrucmed.org/text-unit/",
    "http://digistrucmed.org/normative-assertion/",
    "http://digistrucmed.org/grade-record/",
]


def rescope(graph: Graph) -> int:
    semantic_unit_ids = {str(o) for o in graph.objects(None, DS.semanticUnitId)}
    if not semantic_unit_ids:
        raise ValueError("No ds:semanticUnitId triple found - nothing to scope by.")
    if len(semantic_unit_ids) > 1:
        raise ValueError(
            f"Expected exactly one ds:SemanticUnit per file, found {len(semantic_unit_ids)}: "
            f"{sorted(semantic_unit_ids)}. Run this per-file, before merging."
        )
    scope = next(iter(semantic_unit_ids))
    shared_forms = {o for o in graph.objects(None, DS.surfaceForm) if isinstance(o, URIRef)}

    def rescoped(term):
        if not isinstance(term, URIRef):
            return term
        if term in shared_forms:
            return term
        text = str(term)
        for prefix in RESCOPED_PREFIXES:
            if text.startswith(prefix):
                rest = text[len(prefix):]
                if rest.startswith(scope + "/"):
                    return term  # already scoped - idempotent re-run
                return URIRef(f"{prefix}{scope}/{rest}")
        return term

    triples = list(graph)
    rewritten = 0
    graph_new = Graph()
    for s, p, o in triples:
        new_s, new_o = rescoped(s), rescoped(o)
        if new_s != s or new_o != o:
            rewritten += 1
        graph_new.add((new_s, p, new_o))

    graph.remove((None, None, None))
    for triple in graph_new:
        graph.add(triple)
    return rewritten


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="RDFizer n-triples output to rescope (one source JSON file's worth)")
    parser.add_argument("--output", type=Path, default=None, help="Write here instead of overwriting --path")
    args = parser.parse_args()

    graph = Graph()
    graph.parse(args.path, format="nt")
    before = len(graph)

    rewritten = rescope(graph)

    out_path = args.output or args.path
    graph.serialize(destination=out_path, format="nt", encoding="utf-8")
    print(f"{args.path}: {before} triples, {rewritten} rescoped URI(s) touched -> {out_path}")


if __name__ == "__main__":
    main()

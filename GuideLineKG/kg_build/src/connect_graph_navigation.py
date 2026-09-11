"""Add outgoing paths from Wikidata items back to JSON recommendations.

Run on an existing graph without making any network requests:
    py -3.12 kg_build/src/connect_graph_navigation.py INPUT.nt
Writes INPUT_connected.nt by default; preserves the input.
"""
import argparse
from pathlib import Path
from rdflib import Graph, Namespace, URIRef

DS = Namespace("http://digistrucmed.org/")


def connect_graph_navigation(graph):
    """Materialize reverse links without copying grades to global concepts."""
    additions = Graph()
    for concept, item in graph.subject_objects(DS.wikidataMatch):
        additions.add((item, DS.matchedJsonConcept, concept))
    for mention, concept in graph.subject_objects(DS.surfaceForm):
        if isinstance(concept, URIRef):
            additions.add((concept, DS.hasSourceMention, mention))
    for role in ("Actor", "Condition", "Exception", "Outcome", "Parameter", "Population", "Rationale", "Target"):
        for assertion, mention in graph.subject_objects(DS["has" + role + "Entity"]):
            additions.add((mention, DS.occursInAssertion, assertion))
    before = len(graph)
    graph += additions
    return len(graph) - before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input.with_name(args.input.stem + "_connected.nt")
    if args.input.resolve() == output.resolve():
        parser.error("Choose an output different from the input")
    graph = Graph().parse(args.input, format="nt")
    count = connect_graph_navigation(graph)
    output.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(output, format="nt", encoding="utf-8")
    print(f"Added {count} navigation links; saved {output}")


if __name__ == "__main__":
    main()

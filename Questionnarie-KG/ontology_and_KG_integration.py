"""Ontology and Knowledge Graph integration.

Merges the questionnaire ontology (schema / TBox) with the RML-generated
instance data (ABox) into a single graph and writes the result to the KG/ folder.
"""

from pathlib import Path

from rdflib import Graph

# Resolve paths relative to this script so it runs from any working directory.
BASE_DIR = Path(__file__).resolve().parent

ONTOLOGY_FILE = BASE_DIR / "MappingRules" / "Ontology.ttl"
DATA_FILE = BASE_DIR / "rdf-dump" / "KG-Questionnaire.nt"

OUTPUT_DIR = BASE_DIR / "KG"
OUTPUT_FILE = OUTPUT_DIR / "KG-Questionnaire-with-ontology.nt"

def escape_literal_newlines(text):
    """Repair raw CR/LF characters inside N-Triples quoted literals only."""
    result = []
    state = "outside"
    escaped = False
    for char in text:
        if state == "literal":
            if char in "\r\n":
                result.append("\\r" if char == "\r" else "\\n")
                escaped = False
                continue
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                state = "outside"
        elif state == "iri":
            if char == ">":
                state = "outside"
        elif state == "comment":
            if char in "\r\n":
                state = "outside"
        elif char == '"':
            state = "literal"
        elif char == "<":
            state = "iri"
        elif char == "#":
            state = "comment"
        result.append(char)
    return "".join(result)


def parse_into(graph, path, fmt):
    """Decode legacy dumps and repair literal newlines before parsing."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("cp1252")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
    if fmt == "nt":
        text = escape_literal_newlines(text)
    parsed = Graph()
    parsed.parse(data=text, format=fmt, publicID=path.resolve().as_uri())
    graph += parsed


def main():
    g = Graph()
    parse_into(g, ONTOLOGY_FILE, "turtle")
    parse_into(g, DATA_FILE, "nt")

    # Make sure the output folder exists before serializing.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(OUTPUT_FILE), format="nt", encoding="utf-8")

    print(f"Merged {len(g)} triples -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

# Step 1 — Abbreviation Extraction — `01-abbreviation-extraction/`

Extracts abbreviation glossaries from guideline markdown, merges them across
documents, and converts the result into RDF under the
`http://digistrucmed.org/` namespace. Self-contained — every file this
pipeline needs lives in this folder. Its output, `output/merged_abbreviations.json`,
is a required input to [`../02-wikidata-enrichment/`](../02-wikidata-enrichment/README.md).

Not part of `kg_build/` (the older CSV/flat-mentions pipeline, untouched by
this one) or [`../03-json-knowledge-graph/`](../03-json-knowledge-graph/README.md),
though the RDF this pipeline produces reuses that step's
`ontology/ds-ontology-tables.ttl` `ds:SurfaceForm` class — see
`ontology/abbreviation-ontology.ttl`'s header comment.

## Folder layout

| Folder | Contents |
|---|---|
| `data/input_md/` | Raw source markdown, one subfolder per document (gitignored — large) |
| `ontology/` | `abbreviation-ontology.ttl` — `ds:Abbreviation` and its properties, extends the core ontology in step 3 |
| `mappings/` | `mapping-abbreviations.rml.ttl` — the RML mapping |
| `config/` | `rdfizer-abbreviations.ini` — RDFizer config |
| `src/` | `extract_abbreviations.py`, `merge_abbreviations.py`, `stage_abbreviations.py` |
| `output/` | Intermediate and final pipeline outputs (see stages below) |

## Pipeline stages

```
data/input_md/<document>/page_*.md   (one folder per source document)
            │
            ▼  src/extract_abbreviations.py (per document)
output/per_document/<document>.json   (table / footnote / inline detection,
            │                          deduped, plurals folded)
            ▼  src/merge_abbreviations.py (across all documents)
output/merged_abbreviations.json      (one entry per abbreviation, trust order
            │                          table > footnote > inline)
            ▼  src/stage_abbreviations.py
output/abbreviations.json             (array-of-objects, RML-ready)
            │
            ▼  PYTHONUTF8=1 python -m rdfizer -c config/rdfizer-abbreviations.ini
output/knowledge_graph_abbreviations.nt
```

Run from `01-abbreviation-extraction/`:

```powershell
# 1. Extract per document (one JSON per subfolder of data/input_md)
foreach ($d in Get-ChildItem data\input_md -Directory) {
    py -3.12 src\extract_abbreviations.py $d.FullName -o "output\per_document\$($d.Name).json"
}

# 2. Merge everything in output/per_document into one file
py -3.12 src\merge_abbreviations.py output\per_document -o output\merged_abbreviations.json

# 3. Stage for RML (object -> array) and build the graph
py -3.12 src\stage_abbreviations.py output\merged_abbreviations.json -o output\abbreviations.json
$env:PYTHONUTF8 = "1"
py -3.12 -m rdfizer -c config\rdfizer-abbreviations.ini
```

Both `extract_abbreviations.py` and `merge_abbreviations.py` are idempotent —
re-running after adding a new folder to `data/input_md/` is safe.

## RDF model

`abbreviation`/`expansion`/`plural_forms_seen` are minted as **IRIs** under
`http://digistrucmed.org/entity/{text}`, not literal values, so they can be
linked from the rest of the knowledge graph:

```turtle
ds:entity/VA a ds:Abbreviation, ds:SurfaceForm ;
    rdfs:label "VA" ; ds:name "VA" ;
    ds:abbreviationMethod "footnote" ;
    ds:sourceDocument "lenarczyk-2024-..." ;
    ds:hasExpansion ds:entity/ventricular%20arrhythmia ;
    ds:hasPluralForm ds:entity/VAs .
ds:entity/VAs a ds:Abbreviation, ds:SurfaceForm ;
    ds:pluralFormOf ds:entity/VA .
```

`rr:template` auto-percent-encodes the substituted text into a valid IRI
(spaces etc.), so `hasExpansion` lands on `ds:entity/ventricular%20arrhythmia`
for the readable text `rdfs:label "ventricular arrhythmia"` — the same
`entity/{surface_form}` namespace step 2's and step 3's RML mappings mint
their nodes under.

Identical surface text merges onto one node **across all three steps** once
their `.nt` outputs are loaded together — an abbreviation, an extracted
entity mention, and a Wikidata-sourced concept with the same text share the
same `entity/{text}` IRI, no bridging/`owl:sameAs` step required. See
`ontology/abbreviation-ontology.ttl`'s header comment for details.

## Files

- `src/extract_abbreviations.py` — table/footnote/inline abbreviation-definition
  detection from markdown (see its own docstring for the three detectors and
  the plural-folding rule).
- `src/merge_abbreviations.py` — merges multiple documents' extractions,
  preserving the same table > footnote > inline trust order.
- `src/stage_abbreviations.py` — converts the merged JSON object into the
  array-of-objects shape RML needs.
- `mappings/mapping-abbreviations.rml.ttl` / `config/rdfizer-abbreviations.ini` —
  RML mapping and RDFizer config.
- `ontology/abbreviation-ontology.ttl` — `ds:Abbreviation` and its properties;
  extends `ds:SurfaceForm` from `../03-json-knowledge-graph/ontology/ds-ontology-tables.ttl`.
- `data/input_md/` — source markdown, one subfolder per document.
- `output/per_document/`, `output/merged_abbreviations.json`,
  `output/abbreviations.json`, `output/knowledge_graph_abbreviations.nt` —
  intermediate and final pipeline outputs.

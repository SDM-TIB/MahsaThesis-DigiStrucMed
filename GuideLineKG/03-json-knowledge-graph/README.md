# Step 3 — JSON Knowledge Graph — `03-json-knowledge-graph/`

JSON-direct pipeline that turns stage-4 extraction JSON (`data/level4-dataset/`,
`text_units[]` with `entities[]`, `normative_assertions[]`,
`source_grade_records[]`) into an RDF knowledge graph under
`http://digistrucmed.org/`. This is the core/foundational step — it has no
dependency on steps 1 or 2 and can be run on its own. It also owns the
project's shared ontology (`ontology/ds-ontology-tables.ttl`), which steps 1
and 2 extend/reuse by IRI reference, not by file dependency.

The older, separate CSV/flat-SciSpacy-mentions pipeline lives in `../kg_build/`
and is untouched by this one — see `kg_build/README.md`.

Everything below is run **from `03-json-knowledge-graph/`** unless stated
otherwise.

```powershell
pip install -r ..\kg_build\requirements.txt
```

## Folder layout

| Folder | Contents |
|---|---|
| `data/level4-dataset/` | Small working-set sample (3 files) |
| `data/level4-dataset-full/` | Full raw extraction dataset (539 files, gitignored — large) |
| `ontology/` | `ds-ontology-tables.ttl` — the **core, shared ontology** for all three steps |
| `mappings/` | `mapping-json.rml.ttl` (the RML mapping), `ontology-class-lookup.json` (enum-to-class lookup table the mapping joins against), `level4.json`/`level4_2.json` (RDFizer debug-staging files, see below) |
| `config/` | `rdfizer-json.ini` — RDFizer config |
| `src/` | `run_json_batch.py`, `rescope_json_ids.py`, `enrich_json_types.py`, `normalize_surface_forms.py`, `connect_graph_navigation.py` |
| `output_json/` | `knowledge_graph_json.nt` (merged graph) + `per_unit/` (one `.nt` per source file before merging) |
| `queries/` | `wikidata_to_recommendation_grades.rq` — demo SPARQL query |
| `tests/` | `test_json_class_mapping.py`, `test_normalize_surface_forms.py` |

## Building the graph

**This is the only command you need** to turn every `*.json` file in a
folder into one combined knowledge graph:

```powershell
$env:PYTHONUTF8 = "1"
py -3.12 src/run_json_batch.py data/level4-dataset
```

Everything else in this section is either what that command does under the
hood, or an optional troubleshooting workflow you don't need for normal
use — skip straight to [Reverse-navigation edges](#reverse-navigation-edges-optional)
if the command above already gave you `output_json/knowledge_graph_json.nt`.

### What that command does

For every `*.json` file in the given directory: maps it through
`mappings/mapping-json.rml.ttl` + RDFizer, rescopes local IDs
(`entity_id`/`text_unit_id`/`assertion_id`/`grade_record_id` are only
unique *within* one source file — `src/rescope_json_ids.py` fixes this
before merging), adds value-conditional typing RML can't express
(`src/enrich_json_types.py`). Once every file is done, it folds
cross-document case/whitespace variants of the same surface form
(`src/normalize_surface_forms.py`) and merges everything into
`output_json/knowledge_graph_json.nt`. A file that fails is reported and
skipped, not fatal to the whole batch.

Use `data/level4-dataset` (10 files) for a quick run — seconds, not
minutes, handy for checking the mapping/ontology still work. Swap in
`data/level4-dataset-full` (539 files, gitignored — large) for the real
run: each file is RDFizer-invoked separately (see `process_one()` in
`src/run_json_batch.py`), so this takes noticeably longer and prints one
`file.json: N -> M triples -> ...` line per file as it goes - watch
progress in `output_json/per_unit/`. **The combined
`output_json/knowledge_graph_json.nt` is only (re)written once, at the very
end**, after every file has been attempted — if the batch is interrupted
partway, the per-file `.nt` files already written in `output_json/per_unit/`
are real and usable individually, but the combined file will still be
whatever it was from the last *completed* run, not a partial merge. Re-run
the same command to retry — already-processed files just get overwritten
with the same result.

### Troubleshooting: debugging one failing file in isolation

Only needed if `run_json_batch.py` reported `FAILED` for a specific file
and you want to see that one file's raw RDFizer error directly — it does
**not** process a folder, has no id-rescoping, and its output must not be
merged with anything else:

```powershell
cp data\level4-dataset\0b51cd409b1e_annotation.json mappings\level4_2.json
py -3.12 -m rdfizer -c config/rdfizer-json-debug.ini
py -3.12 src/rescope_json_ids.py output_json/debug_single_file.nt
py -3.12 src/enrich_json_types.py output_json/debug_single_file.nt
```

Replace `data\level4-dataset\0b51cd409b1e_annotation.json` with whichever
file is actually failing. Two things that look interchangeable here but
aren't, both worth getting right:

- **Copy into `mappings\level4_2.json`, not `mappings\level4.json`.** Every
  `rml:source` in `mapping-json.rml.ttl` (which both RDFizer configs below
  use) reads from `mappings/level4_2.json`. `mappings/level4.json` is a
  separate, unrelated fixture the test suite reads directly
  (`tests/test_json_class_mapping.py`) and isn't wired into either RDFizer
  config at all. Copying into the wrong one doesn't error — it just
  silently rdfizes whatever stale content was already sitting in
  `level4_2.json` instead of the file you meant to debug.
- **Use `config/rdfizer-json-debug.ini`, not `config/rdfizer-json.ini`.**
  It writes to `output_json/debug_single_file.nt` instead of
  `output_json/knowledge_graph_json.nt`, so this can't collide with the
  real combined graph even if `cp` fails and the rest of the block runs
  anyway against stale data (PowerShell doesn't stop a pasted block just
  because an earlier line failed — still run these one at a time, not
  pasted together). `output_json/debug_single_file.nt` is throwaway
  scratch output either way; nothing else reads it.

## Reverse-navigation edges (optional)

```powershell
py -3.12 src/connect_graph_navigation.py output_json/knowledge_graph_json_wikidata.nt
```

`knowledge_graph_json_wikidata.nt` is the output of
`../02-wikidata-enrichment/enrich_wikidata.py`'s alternative one-pass
enrichment (see that step's README) run against this step's own
`output_json/knowledge_graph_json.nt`. This script adds `ds:occursInAssertion`
etc. directly into a copy of the graph (writes `..._connected.nt`), so
navigation from a Wikidata match back to its recommendation grade doesn't
need a reasoner. Demo query: `queries/wikidata_to_recommendation_grades.rq`.

**Known pre-existing bug**: the `EntityIn*AssertionMap` RML rules meant to
emit these same edges directly during the graph build currently produce
zero triples (a subjectMap fan-out quirk in the installed RDFizer) — this
script is the working alternative.

## Tests

```powershell
py -3.12 -m pytest tests/ -q
```

## Reference notes

- **IDs are per-file local.** Never concatenate raw `rdfizer` output from
  multiple files directly — always go through `run_json_batch.py`.
- **Grade-record identity**: `grade-record/<unit>/gr_0002` is a
  document-local `ds:SourceGradeRecord` occurrence, not a
  `ds:LevelOfEvidence`/`ds:ClassOfRecommendation` itself — its
  `ds:hasGradeValue` target (e.g. `ds:LevelOfEvidence_B-NR`) is the shared,
  cross-document grade category.
- **Ontology**: `ontology/ds-ontology-tables.ttl`. `entities[].entity_type`
  is a closed enum, each value its own `owl:Class`
  (`rdfs:subClassOf ds:ExtractedEntity`, plus the matching domain class
  where one exists, e.g. `ds:MedicationEntity` ⊑ `ds:Drug`). It also covers
  every class in the project's EER diagram (`Disease`, `Symptom`, `Stage`,
  `Cause`, `Evaluation`, `ScientificResource`, ...) — those currently have
  no RML mapping, because the real extraction JSON has no data for them yet
  (only `{quality, table_assessment, schema_version, text_units, source,
  semantic_unit_id}` exist in practice); the classes are defined ahead of
  data availability, same as `ds:FactualRelation`.
- **Known RDFizer engine limits** (this installed version, verified by
  running it — full detail in `mappings/mapping-json.rml.ttl`'s header
  comment): JSONPath `[?(...)]` filters are silently ignored; a raw JSON
  boolean value crashes the JSON reference resolver (so
  `quality.requires_human_review` etc. aren't mapped, though they remain in
  the ontology); a nested array field needs an explicit `[*]` suffix to fan
  out in `rml:reference`/`rr:template` — a *top-level* array field of the
  current row does not need it in `rr:template`, but **does** need it in a
  plain `rml:reference`.
- **No bare bookkeeping ids/indices in the graph, by design** — this
  matters for anything doing path-based symbolic rule mining over this
  graph (e.g. AMIE/PCA rules for a NeSy chain-of-thought pipeline).
  `entity_id`/`text_unit_id`/`assertion_id`/`grade_record_id` are each
  already the last path segment of their own node's IRI; earlier versions
  of this mapping *also* re-emitted them as a `ds:entityId`/`ds:textUnitId`/
  `ds:assertionId`/`ds:gradeRecordId` triple on that same node. For
  `ds:entityId` specifically this was an `owl:ObjectProperty` from a node
  to ITSELF (`entity-mention/e_0197 ds:entityId entity-mention/e_0197`) —
  a same-node self-loop that is always true by construction, which a rule
  miner will happily turn into a fluent-sounding but empty rule (reproduced
  live: *"if entity id ?b AND source mention ?f, then source mention
  ?b"*, PCA confidence 1.0, over a path built from `e_0197`). Removed, along
  with `locator.cell_index`/`row_index`/`source_part_index`/
  `additional_cell_indices` (a table position number, unlike its
  `header_path` text, carries no meaning on its own). `ds:semanticUnitId`
  is the one id-shaped property still populated — `src/rescope_json_ids.py`
  reads it back out of the graph to scope per-file-local ids before
  merging documents, so it's infrastructure, not reasoning noise.
- **`ds:hasTextUnit` (SemanticUnit → TextUnit) is now actually populated.**
  It was already declared in the ontology but had no RML rule emitting it,
  leaving every `ds:TextUnit` node reachable only going *down* to its own
  locator/entities/assertions/grade records, with no edge back up to the
  guideline it came from. Fixed in `SemanticUnitMap` using the same
  verified top-level-array-of-objects fan-out `TextUnitMap` already relies
  on for `entities[*].entity_id`.

## Current JSON graph identity rules

- `grade-record/<semantic_unit_id>/gr_0002` is a document/source-unit occurrence
  of `ds:SourceGradeRecord`. It is **not** a `ds:LevelOfEvidence` or
  `ds:ClassOfRecommendation`.
- Its `ds:hasGradeValue` target, such as `ds:LevelOfEvidence_B-NR`, is the
  shared grade category and carries that grade class.
- Entity mentions, assertions, text units, grade records, and their references
  are scoped per source unit before merging. Schema classes, surface-form
  resources, entity-type codes, and grade categories retain global IRIs.
- The RML mapping emits the corrected grade-value types directly. The batch
  runner selects each exact input in a temporary mapping, scopes local IDs,
  and merges safely. It does not overwrite the source JSON or mapping.
- Standalone RDFizer output still uses per-file local IDs: do not concatenate
  raw outputs from different files. Use the batch runner below to scope them.

```powershell
py -3.12 kg_build/src/run_json_batch.py kg_build/level4-dataset
```

Example after the multi-document build:

```turtle
<http://digistrucmed.org/grade-record/document_A/gr_0002>
    a ds:SourceGradeRecord ; ds:hasGradeValue ds:LevelOfEvidence_B-NR .
<http://digistrucmed.org/grade-record/document_B/gr_0002>
    a ds:SourceGradeRecord ; ds:hasGradeValue ds:LevelOfEvidence_B-NR .
ds:LevelOfEvidence_B-NR a ds:LevelOfEvidence .
```

The optional compatibility enrichment step removes obsolete grade-category
rdf:types from old source records. Already-merged graphs with colliding local
IDs must be rebuilt from the original JSON files; their lost distinctions
cannot reliably be recovered from the merged graph.

# DigiStrucMed Guideline Knowledge Graph — `kg_build`

Deterministic, traceable, ontology-constrained pipeline that turns SciSpacy/UMLS
prediction JSONL into an RDF knowledge graph under the `http://digistrucmed.org/`
namespace.

Built for this run with:

```text
PREDICTIONS=predictions_compact (1).jsonl   (539 semantic units, 55,640 mentions)
ONTOLOGY=ontology/ds-ontology.ttl
SOURCE_ROOT=NONE                             (extracted_units_corrected/ was not available)
OUTPUT_DIR=kg_build/
```

(Earlier, smaller runs used `prediction_60to63.jsonl` (4 semantic units / 1267
mentions / 55% entity classification) and `prediction_line26.jsonl` (1 semantic
unit / 146 mentions) - see git history / prior report snapshots for those
numbers. The counts throughout this README now reflect the full
`predictions_compact (1).jsonl` run currently in `data/`.

The `data/umls_relations.csv`/`data/umls_semantic_types.csv` enrichment tables
from the `prediction_line26.jsonl` run (25 CUIs) were deliberately dropped
before this run rather than kept stale: they covered a tiny fraction of this
run's 16,865 unique CUIs and would have misrepresented enrichment coverage.
They are currently present only as empty header-only stub files so RDFizer's
mapping can still resolve them; see "UMLS relation & semantic-type enrichment"
below to regenerate them properly for this dataset.)

## Pipeline stages

```
predictions.jsonl ──▶ build_tables.py ──▶ data/*.csv (5 tables)
                                            │
                                            ├──▶ fetch_umls_relations.py ──▶ data/umls_relations.csv
                                            │    (optional, needs UMLS_API_KEY)  data/umls_semantic_types.csv
                                            │
                                            ▼
                                  extract_assertions.py ──▶ data/assertions.csv
                                            │                reports/rejected_or_review.csv
                                            ▼
                                  validate_tables.py ──▶ reports/validation_report.json
                                            │
                                            ▼
                             mappings/mapping.rml.ttl + config/rdfizer.ini
                                            │
                                            ▼
                       PYTHONUTF8=1 python -m rdfizer  ──▶ output/knowledge_graph.nt
                                            │
                                            ▼
                          src/fix_rdfizer_encoding.py (safety net, usually a no-op)
```

`fetch_umls_relations.py` only depends on `data/umls_candidates.csv` (written by
`build_tables.py`), so it can run any time after that and before `validate_tables.py`;
it does not depend on `extract_assertions.py` or vice versa.

### 1. `src/build_tables.py`

Reads `PREDICTIONS` (JSON or JSONL) and writes:

- `data/semantic_units.csv` — one row per prediction record. `source_text` is
  populated by extracting plain text from `SOURCE_ROOT/export_unit_path` when
  that file exists; **in this run `SOURCE_ROOT` was not supplied, so every
  `source_text` is empty** — see "Known limitation: no source text" below.
- `data/mentions.csv` — one row per SciSpacy mention (never dropped, linked or not).
- `data/umls_candidates.csv` — one row per UMLS candidate per mention.
- `data/entities.csv` — the ontology-normalized entity layer. Each mention
  becomes exactly one `ds:` entity, classified by `config/relation_rules.yaml`
  (see "Entity normalization" below). Unclassified entities are kept, never
  dropped, with `ontology_class_uri` empty and `review_status=review_required`.

### 2. `src/fetch_umls_relations.py` (optional, requires `UMLS_API_KEY`)

Enriches the graph with UMLS Metathesaurus facts *about* the CUIs already
linked in `data/umls_candidates.csv` - not guideline-derived, so kept in
separate `ds:UmlsRelation` resources rather than `ds:ExtractionAssertion`.
See "UMLS relation & semantic-type enrichment" below for the full design,
caveats, and setup.

### 3. `src/extract_assertions.py`

Writes `data/assertions.csv` and `reports/rejected_or_review.csv`. See
"Relation extraction and its fallback" below for what is and is not asserted
in a no-source-text run.

### 4. `src/validate_tables.py`

Cross-checks every table against `ontology/ds-ontology.ttl` and against each
other (URI existence, predicate existence, domain/range compatibility, no
empty subject/predicate/object, URI uniqueness, character-offset bounds when
source text exists, accepted UMLS links have CUIs, no `owl:sameAs` for
unreviewed predictions, every assertion has provenance, and - when
`data/umls_relations.csv`/`data/umls_semantic_types.csv` exist - relation
endpoint non-emptiness, `relation_uri` uniqueness, and TUI format). Writes
`reports/validation_report.json` and exits non-zero on any error.

### 5. `mappings/mapping.rml.ttl` + `config/rdfizer.ini`

RML mapping from the CSV tables to RDF, run with SDM-RDFizer. See "Running
RDFizer" below, including a required workaround for two bugs found in the
installed version.

## Running the pipeline

From `kg_build/`:

```bash
python src/build_tables.py --predictions "../predictions_compact (1).jsonl" --ontology "ontology/ds-ontology.ttl" --output-dir "data"

# Optional: requires UMLS_API_KEY (env var or kg_build/.env). Safe to skip -
# every other stage tolerates umls_relations.csv/umls_semantic_types.csv
# being absent.
python src/fetch_umls_relations.py --data-dir "data"

python src/extract_assertions.py --ontology "ontology/ds-ontology.ttl" --rules "config/relation_rules.yaml" --data-dir "data" --reports-dir "reports"

python src/validate_tables.py --ontology "ontology/ds-ontology.ttl" --data-dir "data" --report "reports/validation_report.json"
```

To include source text (once a real `extracted_units_corrected/` directory is
available), add `--source-root "<path>"` to the `build_tables.py` call — this
also unlocks character-offset validation and, in a future revision of
`extract_assertions.py`, real clinical relation extraction (see below).

### Running RDFizer

Installed version: `rdfizer` (SDM-RDFizer) **4.7.5.14**
(`pip show rdfizer`, https://github.com/SDM-TIB/SDM-RDFizer).

```bash
PYTHONUTF8=1 python -m rdfizer -c config/rdfizer.ini
python src/fix_rdfizer_encoding.py output/knowledge_graph.nt
```

**In PowerShell**, `VAR=value command` is bash syntax and silently does
nothing useful there — set the env var as its own statement first:

```powershell
$env:PYTHONUTF8 = "1"
python -m rdfizer -c config/rdfizer.ini
python src/fix_rdfizer_encoding.py output/knowledge_graph.nt
```

(`$env:PYTHONUTF8` persists for the rest of that PowerShell session, so this
only needs to be set once per terminal window.) If you ever see
`UnicodeEncodeError` / `charmap codec can't encode` from `rdfizer`, this is
why — check `$env:PYTHONUTF8` is actually `"1"` before re-running, and note
that RDFizer crashing mid-write leaves `output/knowledge_graph.nt` **stale**
from a previous run, not simply empty; don't trust it until the next run
completes with `Successfully semantified all datasets`.

`config/rdfizer.ini`'s keys (`[datasets]` with `number_of_datasets`,
`all_in_one_file`, `name`, `remove_duplicate`, `enrichment`, `output_folder`,
`output_format`, `ordered`; `[dataset1]` with `name`, `mapping`) were verified
against this installed version's `semantify()` implementation, not guessed.

**Two version-specific issues were found and worked around while getting this
mapping to actually run against 4.7.5.14** (confirmed with isolated
minimal-mapping tests, not assumed from documentation):

1. **`rr:column` produces no triples for CSV/RML sources in this build.**
   Plain R2RML-style `rr:objectMap [ rr:column "..." ]` literal object maps
   are silently skipped entirely. The fix is to use `rml:reference` instead
   (the RML-proper term for non-relational sources) — `mapping.rml.ttl` uses
   `rml:reference` throughout for literal columns, and `rr:template` for any
   URI/IRI object built from a column. `ordered: yes` in `rdfizer.ini` was
   also required — with `ordered: no`, several `rr:template`-based
   `predicateObjectMap`s on shared CSV sources were dropped too.
2. **Output files are written without `encoding="utf-8"`**
   (`rdfizer/__init__.py`'s `open(output_file, "w")`), so on Windows they're
   written in the system codepage (`cp1252` here) instead of UTF-8. With
   `data/umls_relations.csv` present (`relatedIdName` values from
   `has_translation`/`translation_of` UMLS relations can be Cyrillic,
   Japanese, etc.), this isn't just a mis-encoding — `cp1252.encode()` raises
   `UnicodeEncodeError` on characters it has no mapping for at all, and
   `semantify()` **crashes before writing any output.** With only Latin-1-ish
   source text (e.g. the dagger `†`, U+2020), it degrades more quietly to an
   invalid UTF-8 byte sequence that `rdflib` refuses to parse - same root
   cause, different symptom depending on what's in the data that run.

   **Fix: run RDFizer with `PYTHONUTF8=1`** (verified above - this forces
   Python's default text encoding to UTF-8 on Windows, so `open(path, "w")`
   inside `rdfizer` writes correct UTF-8 directly and the crash/mis-encoding
   never happens). `src/fix_rdfizer_encoding.py` is kept as a safety net for
   anyone who forgets to set it: it now checks whether the file already
   decodes as valid UTF-8 and no-ops if so, so it's **safe to always run
   after RDFizer regardless of whether `PYTHONUTF8=1` was set** — it will
   only actually re-encode a genuinely cp1252-written file.

After both steps, `output/knowledge_graph.nt` parses cleanly with `rdflib`
(2,093,360 triples for the current `predictions_compact (1).jsonl` run — no
UMLS-relation enrichment triples in this run, see the enrichment section
below) — verified, not assumed:

```python
from rdflib import Graph
g = Graph()
g.parse("output/knowledge_graph.nt", format="nt")
```

## Entity normalization (`config/relation_rules.yaml`)

Every mention is classified into a `ds:` class (or left unclassified) by two
rule tiers, in this order:

1. **Lexical rules** for guideline-specific vocabulary UMLS can't see:
   `Class I`/`Class IIa` → `ds:ClassOfRecommendation`, `Level A`/`LOE B` →
   `ds:LevelOfEvidence`, "high/low/intermediate/moderate risk" →
   `ds:RiskLevel`, pacing-mode abbreviations (`VVI`, `DDI`, `DDD`, ...) →
   `ds:DeviceMode` (exact allowlist match only — never inferred from context,
   per the abbreviation-ambiguity rule below), the literal header terms
   "Recommendation(s)"/"Guideline(s)", bare recommendation-table tokens
   (`COR`, `IIa`, `IIb`, `III`) → `ds:ClassOfRecommendation`, bare `LOE` →
   `ds:LevelOfEvidence`, HF pharmacotherapy class abbreviations (`ACEI`,
   `ARB`/`ARBs`, `ARNI`/`ARNIs`, `MRA`/`MRAs`, `SGLT2`/`SGLT2i`) → `ds:Drug`,
   ejection-fraction subtype labels (`HFrEF`, `HFpEF`, `HFmEF`, `HFimpEF`) →
   `ds:Type`, and `GDMT` → `ds:Intervention`.
2. **UMLS semantic-type (T-code) rules**, applied to the candidate whose
   `cui == predicted_cui`: `T047`→`ds:Disease`, `T121`/`T195`→`ds:Drug`,
   `T074`→`ds:Device`, `T061`/`T060`/`T059`→`ds:ClinicalProcedure`,
   `T079`→`ds:TemporalConstraint`, `T037`→`ds:SideEffect`, `T184`→`ds:Symptom`,
   `T056`→`ds:Intervention`. Only T-codes that map cleanly onto one ontology
   class are listed; ambiguous ones (`T033` Finding, `T046` Pathologic
   Function, `T078` generic Idea/Concept, `T101` Patient group, `T131`
   Hazardous Substance, etc. — all observed frequently in this dataset) are
   deliberately left unmapped rather than guessed.

On this dataset: **19,593 / 55,640 mentions (35%) classified**, 36,047 left
`review_status=review_required` in `data/entities.csv` (itemized per semantic
unit — 3,736 grouped rows — in `reports/rejected_or_review.csv`).

### Added rules from frequency analysis (this run)

The bare-token/abbreviation lexical rules and the `T184`/`T056`/`T059`
semantic-type rules above were added by analyzing which UMLS semantic types
and surface forms were most common among *unmapped* entities in this specific
539-unit dataset — not guessed from the ontology or UMLS docs alone. Every
addition was checked against `data/mentions.csv`/`data/umls_candidates.csv`
before being written, e.g.: `COR` (114 occurrences) and `IIa`/`IIb` (495/118)
are **100% mislinked** by the SciSpacy/UMLS linker to `Heart` and `Multiple
Endocrine Neoplasia Type 2a/2b` respectively — the same abbreviation-ambiguity
trap as `DDI`→didanosine; `ARNI`→"Arnia" (a plant genus), `ARB`→"Arbitrary
(property)", `SGLT2`→"SLC5A2 gene" are similarly confirmed mislinks; `LOE`,
`HFrEF`/`HFpEF`, and `GDMT` are simply unlinked entirely
(`predicted_cui` is null) rather than mislinked. `T191` (Neoplastic Process,
830 occurrences) was considered and **rejected** as a semantic-type rule: the
majority of hits are `IIa`/`IIb`/`MCS`/`CS` mislinks (genuine neoplasms are
only 16/830), and the `IIa`/`IIb` majority is now correctly intercepted by the
higher-priority `class_of_recommendation_bare_token` lexical rule instead
(lexical rules are checked before semantic-type rules in `classify_entity()`,
[build_tables.py:126-136](src/build_tables.py#L126-L136)).

### Abbreviation ambiguity example encountered in this run

`predicted_cui` sometimes resolves an in-guideline term to an implausible
UMLS concept (e.g. a chemical/element T-code hit on a term that is actually a
guideline abbreviation). Per the skill's rule for this: the SciSpacy
candidate is always preserved in `umls_candidates.csv`; the *local* KG label
comes from `surface_form`, never from `predicted_canonical_name` outright;
and classification only happens via the lexical/semantic-type rules above —
never by trusting a high linker score alone.

## UMLS acceptance policy

Per mention, the candidate matching `predicted_cui` is scored against that
prediction record's own `linker_threshold` (0.7 throughout this dataset):

| Tier | Condition | RDF |
|---|---|---|
| `accepted` | score ≥ `linker_threshold` AND no `review_flags` AND `label_status == predicted_umls_link` | `ds:hasAcceptedUmlsConcept` + `skos:exactMatch` |
| `close_match` | score ≥ 0.85, tier above not met | `skos:closeMatch` only |
| `candidate_only` | everything else | `ds:hasUmlsCandidate` only (no skos relation) |

On this dataset: 39,623 accepted, 6,728 close-match, 9,289 candidate-only.
`owl:sameAs` is never emitted for a linker prediction — nowhere in
`mapping.rml.ttl` is `owl:sameAs` used.

## UMLS relation & semantic-type enrichment (`src/fetch_umls_relations.py`)

**Not run for the current `predictions_compact (1).jsonl` dataset.** This
dataset has 16,865 unique CUIs — fetching MRREL/MRSTY for all of them via the
UTS REST API is a long-running, rate-limited operation, so it was skipped by
default rather than run unattended. `data/umls_relations.csv` and
`data/umls_semantic_types.csv` currently exist only as empty, header-only
stub files (columns below) so RDFizer's mapping can still resolve those
sources; the caveats and stats in this section describe an **earlier, much
smaller run** (25 CUIs) and are kept here as design documentation, not as a
description of what's in `data/` right now. Run the command in "Setup" below
against `data/umls_candidates.csv` to populate this enrichment layer for real
for the current dataset.

Separate from guideline-derived relation extraction below: this stage adds
facts *about* UMLS concepts themselves, pulled from the UMLS Metathesaurus
(MRREL) and semantic-type assignments (MRSTY) via the UTS REST API, for every
CUI already linked in `data/umls_candidates.csv`.

**Setup**: needs a free UMLS Metathesaurus License / UTS API key from
<https://uts.nlm.nih.gov>, provided as `UMLS_API_KEY` — either an environment
variable or a `kg_build/.env` file (loaded automatically via `python-dotenv`).
The key is never logged, printed, or written to any output file; `.env`
should not be committed if this repo is later put under version control.

**Deduplication**: the unique CUI set is computed once from
`umls_candidates.csv` (`set()` over the `cui` column) before any network
call, so a CUI referenced by many mentions/candidates is fetched exactly
once. Each CUI's raw API response is also cached to
`data/umls_cache/{cui}.json`; re-running the script only fetches CUIs that
aren't already cached (pass `--refresh` to force a full refetch).

**Output tables** (URIs precomputed in Python, not built in RML, per this
pipeline's "no string transformation in RML" rule — see `src/build_tables.py`'s
`uri()` helper, reused here):

- `data/umls_relations.csv` — `cui1, cui1_uri, rel, rela, cui2, cui2_uri,
  cui2_name, sab, relation_uri`. `rel`/`rela` are the raw MRREL relation/
  additional-relation labels (e.g. `RO`/`ingredient_of`), `sab` is the source
  vocabulary (e.g. `RXNORM`, `MED-RT`). `relation_uri` is deterministic —
  `ds:umls-relation/{cui1}_{rel}_{rela}_{cui2}_{sab}` — so exact-duplicate
  relations returned by the API collapse to one `ds:UmlsRelation` resource
  via `rdfizer`'s `remove_duplicate: yes`.
- `data/umls_semantic_types.csv` — `cui, cui_uri, tui, semantic_type_name`
  (deduplicated per `(cui, tui)` pair).

**Known data caveats** (found while building this dataset, not assumed):

- `cui2`/`cui2_uri` are taken verbatim from the API's `relatedId` and are
  **not guaranteed to be global UMLS CUIs** — MRREL can return a
  source-vocabulary-local code (e.g. an RXCUI like `331817` for "didanosine
  10 MG/ML") for source-asserted relations. `cui2_uri` is still minted via
  the same `ds:umls/{id}` template either way; that is a valid RDF node
  reference even when nothing else in the graph independently describes it.
- Not every `rela` value is clinically useful for a side-effect/QA use case.
  On this dataset the most frequent values were largely administrative —
  `associated_morphology_of` (637), `has_translation`/`translation_of` (180/177,
  multilingual name pairs — see the RDFizer encoding note above), `inverse_isa`/
  `isa`/`classifies`/`classified_as` (~590 combined, taxonomy structure) — with
  clinically direct ones (`ingredient_of`, `basis_of_strength_substance_of`,
  `precise_active_ingredient_of`) a minority. **No filtering is applied by
  this script** — it fetches and writes everything MRREL returns for a CUI —
  so any downstream consumer (RML mapping, SPARQL queries, fine-tuning data
  prep) that only wants clinically meaningful edges should filter on `rela`
  itself. A curated allowlist (e.g. `may_treat`, `contraindicated_with_disease`,
  `has_ingredient`) would need to be validated against a larger, more
  clinically diverse CUI set than this one (25 CUIs, mostly HIV drug/lab
  content) before being trusted as complete.
- This stage is optional and additive: every other pipeline script tolerates
  `umls_relations.csv`/`umls_semantic_types.csv` being absent (skipped
  entirely if you don't have a UMLS_API_KEY).

**RDF mapping**: `ds:UmlsRelation` follows the same "direct triple + reified
provenance resource" pattern as `ds:ExtractionAssertion` (see Traceability
below) — `ds:umlsRelatedConcept` is the direct shortcut edge between the two
`ds:UMLSConcept` resources, and the corresponding `ds:UmlsRelation` resource
carries `ds:relationLabel`/`ds:relationAttribute`/`ds:relationSource`/
`ds:relationObjectLabel`. It is kept as its own class rather than
`ds:ExtractionAssertion` because its provenance is UMLS itself, not a
guideline `source_text` span — it has no `evidenceText`/`evidenceCharStart`/
`evidenceCharEnd` to report.

## Relation extraction and its fallback (no source text in this run)

`SOURCE_ROOT` was not available for this run (`extracted_units_corrected/`
does not exist on this machine), so `extract_assertions.py` could not see
which entities co-occur in which table row/sentence. Per the skill's explicit
fallback rule, **no clinical relation** (`ds:recommends`, `ds:hasComorbidity`,
`ds:appliesToDevice`, etc.) is fabricated. `data/assertions.csv` contains only
two predicates, both fully determined by the prediction JSON with no
interpretation involved:

- `ds:hasMention` (`SemanticUnit → Mention`) — 55,640 rows
- `ds:derivedEntity` (`Mention → Entity`) — 55,640 rows

Both carry `confidence=1.0` and `extraction_method=structural_provenance`.
Candidate clinical-relation subjects/objects (the 19,593 classified entities,
grouped by semantic unit and class) and the 36,047 unclassified entities are
listed in `reports/rejected_or_review.csv` for human review or for a future
run with `SOURCE_ROOT` supplied. `config/relation_rules.yaml`'s
`clinical_relation_predicates` section documents the domain/range-validated
predicate set `extract_assertions.py` will use once source text is available.

## Traceability

Every row in `data/assertions.csv` produces both the direct triple
(`subject predicate object`) and a `ds:ExtractionAssertion` resource carrying
`ds:assertionSubject/Predicate/Object`, `ds:sourceSemanticUnit`,
`ds:evidenceText`, `ds:evidenceCharStart/End`, `ds:extractionMethod`,
`ds:confidence`, `ds:reviewStatus` — see `<#ExtractionAssertionMap>` in
`mappings/mapping.rml.ttl`. Explicit resources are used instead of RDF-star
for broader RDF-tooling compatibility, per the skill.

## Ontology fields not promoted to RDF

`benchmark_category`, `stage4_relpath`, `model_name`, `linker_name`, and
`umls_source` have no corresponding property in `ontology/ds-ontology.ttl`.
Per "only use predicates defined in the ontology," they remain in
`data/semantic_units.csv` as run provenance but are not mapped to RDF.

## Acceptance test

```bash
python -m pytest tests/test_acceptance.py -v
```

Runs the full pipeline (`build_tables` → `extract_assertions` →
`validate_tables` → RDFizer → `rdflib` parse) against
`tests/fixtures/sample_prediction.jsonl`, one real record copied verbatim
from the supplied `prediction_60to63.jsonl` (`..._n0083`, 256 mentions). All
7 checks pass as of this writing.

`fetch_umls_relations.py` is deliberately **not** part of this test — it
needs live network access and a `UMLS_API_KEY`, so it can't run
deterministically/offline in CI. `validate_tables.py`'s checks for
`umls_relations.csv`/`umls_semantic_types.csv` only activate when those
files exist, so the acceptance test's absence of them is not a failure.

## Setup

```bash
pip install -r requirements.txt
pip install pytest   # only needed to run tests/
```

To run `fetch_umls_relations.py`, also set `UMLS_API_KEY` (free from
<https://uts.nlm.nih.gov>) as an environment variable or in `kg_build/.env` —
see "UMLS relation & semantic-type enrichment" above.

## New approach: JSON-direct pipeline (`mappings/mapping-json.rml.ttl`)

A second, separate pipeline for a different, richer extraction schema
(`schema_version: "1.0"`, e.g. `level4.json` — one semantic unit per file,
with `text_units[]` each carrying `locator`, `entities[]`,
`normative_assertions[]` and `source_grade_records[]`). Everything above this
section (`ontology/ds-ontology.ttl`, `mappings/mapping.rml.ttl`,
`build_tables.py`/`extract_assertions.py`/`validate_tables.py`) is the older
flat-SciSpacy-`mentions[]` pipeline and is untouched by this one.

```text
kg_build/level4-dataset/*.json   (one JSON file per semantic unit)
                                   │
                                   ▼           src/run_json_batch.py loops the
                                   │           steps below once per file:
                                   ▼
                     staged at mappings/level4.json
                                   │
                                   ▼
                    mappings/mapping-json.rml.ttl (ql:JSONPath, reads
                    the JSON directly — no CSV intermediate tables)
                                   │
                                   ▼
              PYTHONUTF8=1 python -m rdfizer -c config/rdfizer-json.ini
                                   │
                                   ▼
      src/rescope_json_ids.py (id-collision fix)  +  src/enrich_json_types.py
      (value-conditional typing RML can't express — see that file's docstring)
                                   │
                                   ▼
       output_json/per_unit/<file>.nt  ──merge──▶  output_json/knowledge_graph_json.nt
```

New classes/properties are defined in `ontology/ds-ontology-tables.ttl`
under "Table/JSON-direct pipeline extensions" — `ds:Page`, `ds:TextUnit`,
`ds:Locator`, `ds:ExtractedEntity`, `ds:EntityNormalization`,
`ds:NormativeAssertion` (subclass of `ds:Recommendation`),
`ds:SourceGradeRecord`, `ds:Population` (previously missing from the
ontology) — covering the drawio diagram's `Guidlines → hasPage → Page →
contains → table → text unit → {locators, factual relations, entities,
normative assertions, source grade records}` branch. The diagram's `Page →
mentions → Entities/candidates` (UMLS-benchmark comparison) subtree is out
of scope and not modeled. `ds:ClassOfRecommendation`/`ds:LevelOfEvidence`
(existing classes) and the new `ds:Population` stay wired into
`ds:NormativeAssertion` exactly as the diagram's `recommend` relation
implies — see `src/enrich_json_types.py`.

`entities[].entity_type` is a closed 20-value enum (`POPULATION`,
`CONDITION_OR_FINDING`, `MEDICATION`, `PROCEDURE_OR_INTERVENTION`,
`TEST_OR_MEASUREMENT`, `DEVICE`, `ANATOMY`, `OUTCOME`, `ADVERSE_EVENT`,
`VALUE`, `TIME`, `DOSE`, `STRENGTH`, `ROUTE`, `FREQUENCY`, `DURATION`,
`FORM`, `ACTOR`, `RATIONALE`, `ENTITY_OTHER`), not a free-form label, so
each value is its own `owl:Class` (`rdfs:subClassOf ds:ExtractedEntity`) in
the ontology's "entityType subclasses" section — not just the
`ds:entityType` string. Where a value is clearly the same concept as an
existing class, its class is *also* `rdfs:subClassOf` that class, e.g.
`ds:MedicationEntity` ⊑ `ds:Drug`, `ds:AdverseEventEntity` ⊑ `ds:SideEffect`,
`ds:DeviceEntity` ⊑ `ds:Device`, `ds:ProcedureOrInterventionEntity` ⊑
`ds:ClinicalProcedure`, `ds:PopulationEntity` ⊑ `ds:Population`,
`ds:ConditionOrFindingEntity` ⊑ `ds:PatientCondition`, `ds:TimeEntity`/
`ds:DurationEntity` ⊑ `ds:TemporalConstraint`. `src/enrich_json_types.py`
materializes both `rdf:type`s directly (this pipeline never assumes an OWL
reasoner runs downstream). `ENTITY_OTHER` (the schema's own catch-all) gets
no dedicated class, same as any future enum value not yet added here.

### Build the knowledge graph from `kg_build/level4-dataset/` (`src/run_json_batch.py`)

This is the primary, supported way to run this pipeline — `kg_build/level4-dataset/`
holds one extraction JSON per semantic unit (11 files as of this writing;
`0b51cd409b1e_annotation.json`, `0b9004ab7668_annotation.json`, ...), and the
batch driver reads every one of them and produces one combined graph:

```bash
PYTHONUTF8=1 python src/run_json_batch.py level4-dataset
```

Run from `kg_build/` (`level4-dataset` here is the relative path to
`kg_build/level4-dataset/`, resolved against that working directory — same
convention as `config/rdfizer.ini`'s `data/...` sources). For each file this
stages it at `mappings/level4.json` (the path baked into
`mappings/mapping-json.rml.ttl` — RML/the installed RDFizer takes one
literal `rml:source` path per run, there is no glob/multi-file input),
invokes RDFizer, then rescopes and enriches the result (see below), writing
`output_json/per_unit/<file stem>.nt`. All per-file graphs are then merged
(deduplicated via `rdflib`) into **`output_json/knowledge_graph_json.nt`** —
the final knowledge graph. A file that fails (e.g. doesn't match the
expected schema) is reported and skipped rather than aborting the whole
batch; the driver exits non-zero if any file failed, after still merging
whatever succeeded.

Verified end-to-end against the current 11-file `kg_build/level4-dataset/`:
all 11 succeed, merging to 13,092 triples — 11 `ds:SemanticUnit`/9
`ds:Guideline` (two files share a guideline), 516 `ds:ExtractedEntity`, 170
`ds:NormativeAssertion` — confirmed stable across repeated runs (identical
triple count both times) and the graph parses cleanly with `rdflib`.

Why a batch driver and not just looping `rdfizer` directly: `entity_id`/
`text_unit_id`/`assertion_id`/`grade_record_id` (e.g. `e_0001`, `na_0001`)
are only unique *within* one source file — every file restarts its own
counters — so naively concatenating multiple files' raw RDFizer output would
silently merge unrelated entities/assertions from different guidelines onto
one URI. `src/rescope_json_ids.py` rewrites those four URI families to embed
the file's own `ds:semanticUnitId` (already globally unique — it embeds the
full `document_sha256`) before merging; `ds:Guideline`/`ds:Page`/
`ds:SemanticUnit` URIs (`document_sha256`/`page_id`/`semantic_unit_id`-keyed
already) don't need this and are left alone. `src/enrich_json_types.py`
then adds the `entityType`/`gradeDimension` value-conditional typing RML
itself can't express (see that file's docstring).

One further Windows-specific fix worth knowing about if you see it again:
`kg_build/`'s working directory sits on a synced/cloud-backed drive, and one
run out of eleven files hit `FileNotFoundError` on the just-written staging
file (`mappings/level4.json`) — the write hadn't propagated before RDFizer's
subprocess tried to open it. `run_json_batch.py` now verifies the staged
file's size before invoking RDFizer and retries the write (up to 5 times,
backing off) if it doesn't match; confirmed fixed by re-running the same
11-file batch twice more with no further failures.

### One file at a time (what the batch driver above does per file)

```bash
cp /path/to/your_unit.json mappings/level4.json
PYTHONUTF8=1 python -m rdfizer -c config/rdfizer-json.ini
python src/rescope_json_ids.py output_json/knowledge_graph_json.nt
python src/enrich_json_types.py output_json/knowledge_graph_json.nt
```

(Also run from `kg_build/`.) Useful for inspecting/debugging one file's
output in isolation; `src/run_json_batch.py` is still the right tool for
anything more than one file, since it also handles the id-rescoping and
merge steps above.

**Known engine limitations found while building this** (verified by
actually running `rdfizer` against `level4.json`, not assumed — see
`mappings/mapping-json.rml.ttl`'s header comment for the full list):
JSONPath `[?(...)]` filter expressions are silently ignored (installed
`jsonpath_ng`, not `jsonpath_ng.ext`); referencing a raw JSON boolean value
crashes `rdfizer`'s JSON reference resolver outright (`table_assessment.*`
booleans and `quality.requires_human_review` are therefore not mapped —
still in the ontology for schema completeness); nested-object array fields
need an explicit `[*]` suffix to fan out and behave differently in
`rr:template` depending on whether they're reached through a parent object
(`source.page_ids[*]` — doesn't fan out in `rr:template`) or through an
array-of-objects (`entities[*].entity_id` — does).


### IRI-valued entity fields (JSON mapping)

`mapping-json.rml.ttl` now emits IRI objects for `ds:entityId`,
`ds:entityType`, and `ds:surfaceForm`. For example (prefixes abbreviated):

```turtle
<http://digistrucmed.org/entity-mention/UNIT/e_0001>
    a ds:ExtractedEntity, ds:AnatomyEntity ;
    ds:entityId <http://digistrucmed.org/entity-mention/UNIT/e_0001> ;
    ds:entityType <http://digistrucmed.org/entity-type/ANATOMY> ;
    ds:surfaceForm <http://digistrucmed.org/entity/Heart> .
<http://digistrucmed.org/entity/Heart> a ds:SurfaceForm ; rdfs:label "Heart" .
```

Occurrence resources use `entity-mention/` so text positions and normalization
remain associated with the original mention. The batch rescoping step adds
`UNIT` to occurrence IDs and their references, while shared surface-form and
entity-type IRIs remain global. Equal surface text shares a lexical node; it
does not imply that ambiguous words denote the same medical concept. RML
itself preserves case and escapes text for use in IRIs (a single standalone
RDFizer run, without the batch runner, still mints one node per exact-cased
string). Labels remain literals.

Two guideline documents rarely spell the same concept identically ("beta
blocker" vs. "Beta Blocker"), and RML has no case-folding function in this
RDFizer build to catch that within one mapping run. `src/normalize_surface_forms.py`
runs after `run_json_batch.py` merges every document's graph and folds
`ds:SurfaceForm` nodes onto one canonical, casefolded/whitespace-collapsed IRI
per concept, moving every triple (labels, `ds:hasSourceMention`, and any
`ds:wikidataMatch` added later by `enrich_wikidata.py`) onto that shared node.
All original-cased labels are kept as separate `rdfs:label` values; only the
IRI is canonicalized. This runs once, on the merged graph, since the
duplication this catches is cross-document by nature.

The corresponding ontology is `ontology/ds-ontology-tables.ttl`; the older
CSV pipeline and its separate ontology keep their existing literal model.
`../enrich_wikidata.py` accepts both legacy literal surface forms and these
IRI-valued surface forms, decoding the text after `/entity/` as query text
(with `rdfs:label` as a fallback for other IRI namespaces). It searches labels/aliases in English and German (up to five candidates each),
selects the best lexical match, and retrieves P31, P279, P1542, P2579, P2176,
P2175, and P715 via SPARQL. All available direct values are kept, together
with English/German labels, aliases, and descriptions for the selected item
and its directly related items. Related items are not recursively expanded.
Local predicates use the names in `PROPERTIES` (for example `ds:instance_of`
and `ds:drugbank_id`); local resource IRIs use `digistrucmed.org/wikidata/`.
Source links retain the original Wikidata IRIs. The JSON report records the
selected QID, search label, language, rank, and lexical similarity per term.
A top lexical match is a candidate, not a verified medical interpretation.
Missing properties or translations are left absent. Run
`py -3.12 enrich_wikidata.py` from the project root to save an enriched copy.

To regenerate the JSON graph from the project root:

```powershell
py -3.12 kg_build/src/run_json_batch.py kg_build/level4-dataset
```

Changing the mapping does not migrate an already generated `.nt` file.


#### Wikidata timeout recovery

`enrich_wikidata.py` defaults to `--source auto`: SPARQL gets one attempt
with a 25-second read timeout. On failure the script switches to
`wbgetentities` for the current item and all subsequent items. The API path
retrieves the same seven properties and English/German metadata, using
preferred statements when available, otherwise normal statements; deprecated
statements are excluded. The report records each item's `data_source` and
`sparql_fallback_reason`. To bypass the SPARQL service immediately:

```powershell
py -3.12 enrich_wikidata.py --source api
```

An already running process must be stopped and restarted to load this change.
The existing input graph is preserved; the enriched output is written at the
end of the run. `--source sparql` remains available for SPARQL-only execution.


### Ontology classes directly in the JSON RML mapping

The mapping now uses `mappings/ontology-class-lookup.json` to translate
`grade_dimension` and `entity_type` values into existing ontology class IRIs.
For example, `B-NR` produces a shared value typed `ds:LevelOfEvidence` (the source
record remains only `ds:SourceGradeRecord`), and
`MEDICATION` produces `rdf:type ds:MedicationEntity` and `rdf:type ds:Drug`.
The original grade dimension and raw grade remain literals as required by
`ds-ontology-tables.ttl`. An extracted grade is an instance of its grade class;
its value (such as "B") is not a new OWL class.

This supersedes the earlier notes saying type assignment requires Python.
RML lookup maps also emit `owl:Class` declarations for the referenced classes.
The lookup is a small explicit enum mapping, not a replacement ontology.
Keep it aligned with the ontology and the compatibility enrichment script.
The enrichment step still adds assertion-to-grade shortcut edges.

Regenerate using the source currently selected in the mapping, from `kg_build/`:

```powershell
$env:PYTHONUTF8 = "1"
py -3.12 -m rdfizer -c config/rdfizer-json.ini
py -3.12 src/enrich_json_types.py output_json/knowledge_graph_json.nt
```

The raw-RDFizer regression test checks every entity type, both grade classes,
unknown values, and that all class IRIs are declared in the ontology:

```powershell
py -3.12 -m unittest discover -s kg_build/tests -p test_json_class_mapping.py
```


### Shared recommendation classes and evidence levels

Repeated source grades now link to one global grade resource:

```turtle
ds:ClassOfRecommendation_I a ds:ClassOfRecommendation ; rdfs:label "I" .
ds:LevelOfEvidence_B-NR a ds:LevelOfEvidence ; rdfs:label "B-NR" .
<grade-record/document1/gr1> ds:hasGradeValue ds:ClassOfRecommendation_I .
<grade-record/document2/gr2> ds:hasGradeValue ds:ClassOfRecommendation_I .
<assertion/document1/a1> ds:hasClassOfRecommendation ds:ClassOfRecommendation_I .
```

These are shared grade categories (named individuals of the existing ontology
classes). Their labels are defined once, rather than on each source record.
Source records retain rawGrade, gradeDimension, grading system, and provenance;
assertions retain hasSourceGradeRecord. Shared IRIs identify dimension plus
exact source grade text, not document IDs. This groups identical grade codes;
it does not assert equivalence between grading systems. `1` and `I` stay distinct.

RML defines the shared grades listed in ontology-class-lookup.json. The
enrichment step links records through hasGradeValue and handles values for the two
recognized dimensions and redirects assertion grade shortcuts to shared nodes.
Unknown dimensions are preserved without guessing. Run the enrichment step
after RDFizer as before. To migrate an existing graph while preserving its file:

```powershell
py -3.12 kg_build/src/enrich_json_types.py kg_build/output_json/knowledge_graph_json.nt --output kg_build/output_json/knowledge_graph_shared_grades.nt
```


### Shared grade links generated by RML itself

`mapping-json.rml.ttl` now emits `ds:hasGradeValue` directly with a
`rr:parentTriplesMap` join to the controlled `grade_values` vocabulary.
For example, source records with `raw_grade: "I"` all link to
`ds:ClassOfRecommendation_I`; `"B-NR"` links to `ds:LevelOfEvidence_B-NR`.
Both the shared resource's definition and each record's link are present in
raw RDFizer output. No Python enrichment or reasoner is needed for those links.

The vocabulary uses unique raw codes across the two grade categories; do not
add the same code under two different categories. The raw code determines the
shared category. Original gradeDimension metadata is retained independently.
Unknown codes remain rawGrade literals until explicitly added to the vocabulary
and ontology. Recommendations reach the shared grade via
`hasSourceGradeRecord / hasGradeValue`. The optional enrichment script also adds
direct recommendation shortcuts, but is not required for the shared-grade model.

Run from `kg_build/`:

```powershell
$env:PYTHONUTF8 = "1"
py -3.12 -m rdfizer -c config/rdfizer-json.ini
```


### Wikidata: follow treatment links to drug details

`enrich_wikidata.py` now performs an additional step after selecting the item:

1. Read outgoing P2176 treatment links.
2. Search for incoming P2175 links with `haswbstatement:P2175=Q...` through
   Wikidata's search API; verify those links against the returned item claims.
3. Visit each related treatment through the entity API and retrieve DrugBank
   ID, ATC code, administration route, significant drug interactions, chemical
   formula, CAS number, PubChem CID, roles, and the original seven properties.
4. Add English/German labels, aliases, and descriptions, retaining Wikidata
   source links and the actual direction of each treatment relationship.

Example graph path: surfaceForm -> wikidataMatch ->
drug_or_therapy_used_for_treatment -> drugbank_id. Reverse matches have the
medical_condition_treated edge from the drug to the condition instead.
DrugBank IDs belong to the drug node, not the condition node.

```powershell
py -3.12 enrich_wikidata.py --source api --max-related-treatments 30
```

Treatment expansion uses the entity/search APIs in all source modes. The
limit bounds visits to related items per match. The report records visited
QIDs, bilingual labels, DrugBank IDs, failures, and truncation. A missing ID
is left absent. P2176 also includes procedures/therapies, so an item is not
automatically asserted to be a drug. Interactions and other related entities
receive metadata but are not recursively expanded. Successful facts are saved
even if a related-item request fails; such partial failures give exit code 1.

Property reference: https://www.wikidata.org/wiki/Property:P2176
Reverse-search syntax: https://www.mediawiki.org/wiki/Help:Extension:WikibaseCirrusSearch


Drug classes now expand to individual drugs as well: the search API finds
items with P31, P279, or P2868 pointing to the class and with a P715 identifier.
Each relationship is checked against the item's current selected-rank claims.
The output preserves the actual class/role link rather than inventing a
condition-treatment relationship. The report records `drugbank_identifiers`,
and the console prints their count. In N-Triples, search for the full predicate
`http://digistrucmed.org/drugbank_id` (prefix declarations are not used).
Scientific-article search hits are excluded by their English/German descriptions;
other ambiguous matches still require review in `selected_matches`.


### Shared clinical types and readable Wikidata attributes

The RML `ClinicalConceptClassLookupMap` types shared `/entity/{surface_form}`
nodes using `entity_concept_types` in ontology-class-lookup.json. Examples:
MEDICATION -> Drug, ADVERSE_EVENT -> SideEffect, POPULATION -> Population.
The occurrence still has its document-scoped ExtractedEntity/MedicationEntity
classification; the shared concept and imported Wikidata item are not mentions.

Wikidata enrichment uses the same lookup. A matched item receives the clinical
classification of its JSON concept and a `classificationSource` link explaining
where that type came from. This is extraction-based classification of a candidate
match, not independent identity validation. All retrieved items with a nonempty
DrugBank ID are explicitly typed Drug. Other related nodes are not assigned the
parent's type blindly; they retain their Wikidata relationships and identity.

Every retrieved label is written as `rdfs:label` and the compatible `ds:label`.
A preferred English (otherwise German) label is stored as `ds:name`. For drugs,
`ds:drugName` is also present. `ds:wikidataId` stores the QID separately, and
`ds:wikidataSource` preserves the full original IRI. Example:

```turtle
<http://digistrucmed.org/wikidata/entity/Q412515>
    a ds:Drug, ds:WikidataEntity ;
    rdfs:label "bisoprolol"@en ;
    ds:name "bisoprolol" ;
    ds:wikidataId "Q412515" ;
    ds:drugBankId "DB00612" .
```

Names and identifiers are RDF literal **attributes** of the same node. Set the
viewer node caption to `rdfs:label` or `ds:name` to display ?bisoprolol? rather
than the IRI suffix. QID-based IRIs remain stable even when a label changes and
prevent different items with the same name from being merged.

Rebuild the JSON graph with the batch runner, then run `enrich_wikidata.py`.
Older graphs with entityType on mentions are also supported by the enricher.


### Navigate from Wikidata back to recommendation grades

Outgoing navigation now follows:

`Wikidata item --matchedJsonConcept--> JSON concept --hasSourceMention-->
document mention --occursInAssertion--> recommendation --hasSourceGradeRecord-->
source grade record --hasGradeValue--> shared grade category`.

The RML mapping emits hasSourceMention and occursInAssertion; the ontology
declares their inverse relationships. To materialize all three reverse links
in an existing enriched graph without querying Wikidata:

```powershell
py -3.12 kg_build/src/connect_graph_navigation.py kg_build/output_json/knowledge_graph_json_wikidata.nt
```

This writes `knowledge_graph_json_wikidata_connected.nt`, preserving the input.
The query `queries/wikidata_to_recommendation_grades.rq` demonstrates navigation
from beta blocker (Q816759). Grades stay on the document-specific assertion:
they are not asserted globally on every drug in a class.

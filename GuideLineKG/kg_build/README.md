> This is the **older, CSV-based flat-SciSpacy-mentions pipeline** only. The
> newer, separate JSON-direct pipeline (stage-4 extraction JSON → RDF, plus
> Wikidata enrichment) has moved to [`Stage4/README.md`](../Stage4/README.md).
> Abbreviation-glossary extraction/RDF has moved to
> [`Abbrevation/`](../Abbrevation/). Neither of those touches this pipeline
> or vice versa.
>
> **Known pre-existing gap, not introduced by that move**: `tests/test_acceptance.py`
> expects `kg_build/ontology/ds-ontology.ttl` and `mappings/mapping.rml.ttl`
> for this CSV pipeline; neither currently exists (only a stale duplicate at
> the old root `ontology/ds-ontology.ttl` did, now removed as dead weight —
> it was never the file the test actually resolved to). Reconstructing this
> pipeline's own RML mapping/ontology copy is a separate task from file
> reorganization.

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


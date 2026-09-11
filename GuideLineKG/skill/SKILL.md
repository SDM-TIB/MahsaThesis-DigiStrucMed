---
name: guideline-kg-builder
description: Build a simple, deterministic, traceable clinical-guideline knowledge-graph pipeline from SciSpacy JSON/JSONL predictions and a DigiStrucMed ontology. Generates normalized CSV tables, ontology-constrained assertions, RML mappings, an SDM-RDFizer config, validation checks, and UMLS links.
---

# Guideline KG Builder

## Purpose

Use this skill when the user provides SciSpacy/UMLS prediction JSON or JSONL plus the DigiStrucMed ontology and wants code that turns the data into RDF.

The pipeline must remain:

- simple enough to inspect;
- deterministic where possible;
- traceable from every RDF assertion back to the source semantic unit and text span;
- ontology-constrained;
- conservative about UMLS links;
- runnable with SDM-RDFizer/RDFizer;
- easy to extend later.

The project namespace is always:

```turtle
@prefix ds: <http://digistrucmed.org/> .
```

Never change this base namespace unless the user explicitly asks.

## Four-line interface

The user should only need to provide these four lines:

```text
PREDICTIONS=<path/to/predictions.jsonl-or-json>
ONTOLOGY=<path/to/ds-ontology.ttl>
SOURCE_ROOT=<path/to/extracted/source/units-or-NONE>
OUTPUT_DIR=<path/to/output/project>
```

If the user pastes 1–4 example prediction records instead of a path, use them to understand the schema, but still create code that works on the full JSON/JSONL file.

Do not ask follow-up questions if paths and files are inspectable. Inspect them first.

## Expected input shape

SciSpacy prediction records may contain fields such as:

- `schema_version`
- `semantic_unit_id`
- `document_sha256`
- `benchmark_category`
- `export_unit_path`
- `stage4_relpath`
- `text_sha256`
- `n_mentions`
- `n_linked_mentions`
- `mentions[]`
- `model_name`
- `linker_name`
- `linker_threshold`
- `umls_source`
- `review_status`

A mention may contain:

- `surface_form`
- `char_start`
- `char_end`
- `ner_label`
- `predicted_cui`
- `predicted_score`
- `predicted_canonical_name`
- `candidates[]`
- `review_flags[]`

Do not assume that a high SciSpacy score means the UMLS concept is contextually correct. Abbreviations such as MRT, HSM, DDI, or ICD can be mislinked. Preserve the original prediction as provenance/candidate data, but do not silently promote it to a verified semantic identity.

## Output project

Create this structure:

```text
OUTPUT_DIR/
  README.md
  requirements.txt
  src/
    build_tables.py
    extract_assertions.py
    validate_tables.py
  config/
    relation_rules.yaml
    rdfizer.ini
  mappings/
    mapping.rml.ttl
  data/
    semantic_units.csv
    mentions.csv
    umls_candidates.csv
    entities.csv
    assertions.csv
  ontology/
    ds-ontology.ttl
  output/
    knowledge_graph.nt
  reports/
    validation_report.json
    rejected_or_review.csv
```

If some files cannot yet be generated because source text is unavailable, still generate the code and clearly mark the missing runtime prerequisite in `README.md`.

## Workflow

### 1. Inspect before coding

Read:

1. the ontology;
2. at least four representative prediction records when available;
3. source text/HTML for those records when `SOURCE_ROOT` is provided.

Summarize the observed schema internally before writing code.

Do not invent source fields that do not exist.

### 2. Normalize predictions into CSV tables

Generate `src/build_tables.py`.

It must support both JSON and JSONL.

Use UTF-8 everywhere.

Create these tables.

#### `semantic_units.csv`

Required columns:

```text
semantic_unit_uri
semantic_unit_id
document_sha256
text_sha256
benchmark_category
export_unit_path
stage4_relpath
model_name
linker_name
umls_source
review_status
source_text
```

`semantic_unit_uri` must be deterministic, e.g.:

```text
http://digistrucmed.org/semantic-unit/{urlencoded-semantic_unit_id}
```

If source text is available from `export_unit_path`, extract plain text conservatively and save it in `source_text`.

#### `mentions.csv`

Required columns:

```text
mention_uri
semantic_unit_uri
surface_form
char_start
char_end
ner_label
predicted_cui
predicted_score
predicted_canonical_name
label_status
gold_status
review_flags
```

Create deterministic mention URIs from `semantic_unit_id`, `char_start`, and `char_end`.

Never discard unlinked mentions.

#### `umls_candidates.csv`

One row per candidate:

```text
candidate_uri
mention_uri
cui
score
canonical_name
types
candidate_rank
umls_concept_uri
umls_browser_url
```

Use:

```text
umls_concept_uri = http://digistrucmed.org/umls/{CUI}
umls_browser_url = https://uts.nlm.nih.gov/uts/umls/concept/{CUI}
```

#### `entities.csv`

This is the ontology-normalized entity layer:

```text
entity_uri
label
normalized_label
ontology_class_uri
source_mention_uri
semantic_unit_uri
normalization_method
normalization_confidence
review_status
accepted_umls_cui
```

Use the ontology to assign only classes that actually exist.

Preserve an entity even when no UMLS CUI is accepted.

Do not use SciSpacy canonical names as the entity label when the context contradicts them.

#### `assertions.csv`

This is the most important table:

```text
assertion_uri
subject_uri
predicate_uri
object_uri
semantic_unit_uri
evidence_text
evidence_char_start
evidence_char_end
extraction_method
confidence
review_status
```

Every KG edge must appear here first.

Only use predicates defined in the ontology.

If no defensible relation can be extracted, do not invent one. Put the candidate item in `reports/rejected_or_review.csv`.

### 3. Ontology-constrained entity mapping

Read all `owl:Class`, `owl:ObjectProperty`, and `owl:DatatypeProperty` definitions from the ontology.

Create a small explicit mapping layer in `config/relation_rules.yaml`.

Prefer deterministic rules based on:

- table headers;
- section labels;
- lexical patterns;
- nearby entities;
- repeated row/column structure;
- known domain abbreviations;
- ontology domain/range constraints.

Do not use unrestricted free-form predicates.

The initial ontology supports classes such as:

- `ds:Guideline`
- `ds:Recommendation`
- `ds:ClassOfRecommendation`
- `ds:LevelOfEvidence`
- `ds:ScientificResource`
- `ds:Comorbidity`
- `ds:Intervention`
- `ds:Device`
- `ds:Drug`
- `ds:ClinicalProcedure`
- `ds:SideEffect`
- `ds:Disease`
- `ds:Type`
- `ds:Stage`
- `ds:Symptom`
- `ds:Cause`
- `ds:Evaluation`

and extensions for guideline tables:

- `ds:PatientCondition`
- `ds:DeviceMode`
- `ds:RiskLevel`
- `ds:TemporalConstraint`
- `ds:SemanticUnit`
- `ds:Mention`
- `ds:UMLSConcept`
- `ds:ExtractionAssertion`

Use subclass relationships when appropriate.

### 4. Relation extraction

Generate `src/extract_assertions.py`.

The first version should be transparent and rule-based rather than opaque.

It may use `relation_rules.yaml` and source layout/text.

Preferred pattern:

```text
source structure -> entity classes -> allowed predicate -> domain/range check -> assertion
```

Examples of allowed relations include:

- `ds:containsRecommendation`
- `ds:recommends`
- `ds:hasClassOfRecommendation`
- `ds:hasLevelOfEvidence`
- `ds:hasComorbidity`
- `ds:hasSideEffect`
- `ds:diagnosedBy`
- `ds:definedBy`
- `ds:hasStage`
- `ds:hasSymptom`
- `ds:hasCause`
- `ds:hasEvaluation`
- `ds:supportedBy`
- `ds:appliesToDevice`
- `ds:hasDeviceMode`
- `ds:hasPatientCondition`
- `ds:hasRiskLevel`
- `ds:hasTemporalConstraint`

For each extracted assertion, save exact evidence text and source semantic unit.

If source text is absent, relation extraction must fall back to conservative mention-level output and mark relations as `review_required`; it must not fabricate relational meaning.

### 5. UMLS integration

UMLS is an enrichment layer, not the source of truth for local guideline semantics.

For every predicted CUI:

1. preserve the SciSpacy candidate;
2. create a local `ds:UMLSConcept` node;
3. add `ds:umlsCui`;
4. add `rdfs:label` from the SciSpacy canonical name when present;
5. add `rdfs:seeAlso` to the UMLS browser URL;
6. connect the mention with `ds:hasUmlsCandidate`.

Only create `ds:hasAcceptedUmlsConcept` when the link is accepted by deterministic validation or human review.

Do not use `owl:sameAs` for an unreviewed linker prediction.

For accepted, strong equivalence use `skos:exactMatch`; otherwise prefer `skos:closeMatch` or only local provenance properties.

### 6. Traceability model

Every clinical triple must be traceable.

For each row in `assertions.csv`, RDF must include both:

```text
subject predicate object
```

and an `ds:ExtractionAssertion` resource with:

- `ds:assertionSubject`
- `ds:assertionPredicate`
- `ds:assertionObject`
- `ds:sourceSemanticUnit`
- `ds:evidenceText`
- `ds:evidenceCharStart`
- `ds:evidenceCharEnd`
- `ds:extractionMethod`
- `ds:confidence`
- `ds:reviewStatus`

Use explicit assertion resources instead of RDF-star so the output works with more RDF tooling.

### 7. RML mapping

Generate `mappings/mapping.rml.ttl`.

Keep the mapping simple.

Do not perform complicated transformations in RML. Precompute stable URIs and cleaned fields in Python, then map CSV columns directly.

Create TriplesMaps for:

- semantic units;
- mentions;
- UMLS concepts/candidates;
- normalized entities;
- assertions;
- direct clinical triples.

Use only standard RML/R2RML terms unless the installed RDFizer requires something else.

All generated resources must use the `http://digistrucmed.org/` namespace except external UMLS browser links.

### 8. RDFizer config

Generate `config/rdfizer.ini` for the installed SDM-RDFizer/RDFizer version.

Before writing the final config:

1. check whether an RDFizer executable/package is installed;
2. inspect its version or local examples/help when possible;
3. generate config keys matching that version.

Do not guess unsupported configuration keys.

If RDFizer is not installed, create a conservative template and label it `UNVERIFIED_TEMPLATE` in the README.

### 9. Validation

Generate `src/validate_tables.py`.

Validation must check:

- every `subject_uri` and `object_uri` exists or is an allowed external URI;
- every `predicate_uri` exists in the ontology;
- ontology domain/range compatibility where declared;
- no empty subject/predicate/object in assertions;
- deterministic URI uniqueness;
- character offsets are valid when source text exists;
- accepted UMLS links have CUIs;
- unreviewed UMLS predictions are not emitted as `owl:sameAs`;
- every direct clinical triple has a matching assertion provenance node.

Write results to `reports/validation_report.json`.

### 10. Run the pipeline

Create commands in README similar to:

```bash
python src/build_tables.py \
  --predictions "<PREDICTIONS>" \
  --source-root "<SOURCE_ROOT>" \
  --ontology "ontology/ds-ontology.ttl" \
  --output-dir "data"

python src/extract_assertions.py \
  --ontology "ontology/ds-ontology.ttl" \
  --rules "config/relation_rules.yaml" \
  --data-dir "data"

python src/validate_tables.py \
  --ontology "ontology/ds-ontology.ttl" \
  --data-dir "data" \
  --report "reports/validation_report.json"
```

Then add the exact RDFizer command appropriate to the installed version.

If RDFizer can be run in the environment, run it and validate the produced RDF using RDFLib.

### 11. Acceptance test

Use at least one supplied prediction record as an end-to-end test.

The test must demonstrate:

- a semantic unit;
- at least one mention;
- at least one UMLS candidate;
- at least one ontology-normalized local entity;
- at least one traceable assertion when supported by the source;
- generated RDF parses successfully.

Never fake a clinical assertion merely to satisfy the acceptance test.

## Coding requirements

- Python 3.10+.
- Prefer standard library + `pandas`, `rdflib`, `PyYAML`, `beautifulsoup4`.
- Small functions with type hints.
- No database required.
- No hidden network calls.
- UMLS web/API enrichment must be optional.
- Never require an API key for the basic pipeline.
- Deterministic IDs.
- UTF-8.
- Log counts at every stage.
- Preserve raw input.
- Do not overwrite source files.
- Put questionable mappings in review output rather than silently accepting them.

## Rule for ambiguous clinical abbreviations

If surface form, source context, and SciSpacy canonical name conflict:

1. preserve the SciSpacy candidate;
2. prefer local/contextual entity semantics for the KG;
3. mark the UMLS link as candidate/rejected/review-required;
4. never replace the local label with an implausible UMLS canonical name.

Example pattern:

```text
surface form: DDI
SciSpacy candidate: didanosine
context: pacemaker/device mode table
```

Represent `DDI` as a local `ds:DeviceMode` and keep the didanosine CUI only as a rejected or unverified candidate.

## Definition of done

The task is complete only when the output contains:

1. runnable table-building code;
2. ontology-constrained assertion extraction code;
3. CSV tables;
4. RML mapping;
5. RDFizer config;
6. extended ontology using `ds:`;
7. validation script/report;
8. README with exact commands;
9. UMLS candidate/accepted-link handling;
10. traceable RDF provenance.

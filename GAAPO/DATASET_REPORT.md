# English dataset and smoke-test changes

The active `GAAPO/data/train.json`, `validation.json` and `test.json` now use
user-provided ChatGPT exchanges from:

- `Chat GPT Chats v.a.M1.docx`
- `Chat GPT Projekte M1-3.docx`
- `Chat GPT Projekte M4-6.docx`

All 347 exchanges were extracted. A curated subset of 28 patient-education
exchanges was translated and condensed into English one-scene request/output
pairs. This is not a literal translation of every chat or a conversion of every
follow-up edit into an independent example. The complete extraction is kept
in `data/document_import/source_exchanges.json`.

| Split | Examples | Source modules |
| --- | ---: | --- |
| Train | 16 | 1, 2, 4 |
| Validation | 4 | 3 |
| Test | 8 | 5, 6 |

Whole source modules remain together to keep closely related conversation
revisions within one split. IDs and normalized English requests were checked
for overlap. This module-level division measures transfer between topics;
it is not an iid benchmark or a guarantee of semantic independence.

Each reference has one scene and at most 35 narration words. Provenance
includes the source document, paragraph number, original German exchange,
translated original request, English response summary and adaptation notes.
The old invented fixtures are backed up in `data/starter_backup/`.

These are **unreviewed ChatGPT-derived reference drafts**. Their medical
claims, source quotations and translation fidelity have not been certified
by a clinician; the originally cited PDFs were not provided. They support
pipeline development rather than claims of clinical validity.

The previous all-zero smoke result came from strict JSON-only scoring of
Markdown scripts and wrapped JSON. The local configuration now uses an
explicit JSON schema, shorter requests and a 256-token output budget. The
separate task evaluator combines strict JSON validity (weight 0.5) with
structural completeness (weight 0.5). Strict validity remains zero for
wrapped JSON. Plain-text structure receives capped partial credit. Every
call logs parsing and schema errors without silently repairing the output.

GAAPO's five generation methods, scalar-fitness interface, evaluation loop
and ranked elitist selection are unchanged. A score of 1 means the output
has the requested JSON fields; it does not establish medical correctness,
coverage, or compliance with the requested one-scene count.

Validation: 31 unit tests passed; all 28 references passed schema, brevity
and split checks. A mock run using the actual English datasets completed.
The real nine-call run and its intermediate artifacts are in
`GAAPO/runs/document-smoke/`.

The real run completed. Validation fitness increased from 0.05 for the initial
prompt to 1.00 for the APO candidate. Its JSON was structurally valid, although
it contained two scenes rather than the requested one. On the one held-out test
request, baseline and optimized fitness were both 0.05 and neither output was
strictly valid JSON. The pipeline/scoring changes therefore improved validation
feedback but did not demonstrate generalization or clinical improvement.
Readable per-call files are in `runs/document-smoke/storyboards/`; the summary
is `runs/document-smoke/review.md`.

Run again in a fresh output directory:

```powershell
python -m GAAPO.experiment --config GAAPO/config.local.json --output GAAPO/runs/document-smoke-next
```

# Step 2 — Wikidata Enrichment — `02-wikidata-enrichment/`

Sources structured Wikidata facts (drug class, evidence, related treatments,
external IDs) about the entities extracted by the JSON pipeline, and mints
them under the **same** `http://digistrucmed.org/entity/{text}` namespace
that pipeline and step 1's abbreviation pipeline use — so a Wikidata-sourced
concept and a same-text extracted entity mention or abbreviation land on the
same RDF node once all three steps' outputs are loaded together.

Depends on: [`../01-abbreviation-extraction/`](../01-abbreviation-extraction/README.md)'s
`output/merged_abbreviations.json`, and the raw extraction JSON in
[`../03-json-knowledge-graph/data/`](../03-json-knowledge-graph/README.md)
(this step only *reads* that data — it does not need step 3's graph to have
been built first).

Everything below is run **from `02-wikidata-enrichment/`** unless stated
otherwise.

```powershell
pip install -r ..\kg_build\requirements.txt
```

## Folder layout

| Folder | Contents |
|---|---|
| `mappings/` | `mapping-wikidata-entities.rml.ttl` — the RML mapping |
| `config/` | `rdfizer-wikidata-entities.ini` — RDFizer config |
| `src/` | `prepare_wikidata_search.py`, `search_wikidata_candidates.py`, `expand_wikidata_entities.py` |
| `enrich_wikidata.py` | Legacy one-pass alternative (see below) — stays a sibling of `src/`, not inside it, because `enrich_wikidata.py` predates `src/` and both `search_wikidata_candidates.py`/`expand_wikidata_entities.py` still import it via a fixed relative path |
| `wikidata_search/` | Generated intermediate/final outputs (see stages below) |
| `tests/` | `test_enrich_wikidata.py` |

No `ontology/` folder here: this step only uses classes already defined in
`../03-json-knowledge-graph/ontology/ds-ontology-tables.ttl` (`ds:WikidataEntity`,
`ds:Drug`, ...), referenced by IRI — it doesn't need its own copy.

## Pipeline stages

Three stages, run in order. Each reads the previous stage's output file and
can be re-run on its own once its input exists.

### a. Preprocess extracted entities into Wikidata search candidates

```powershell
py -3.12 src/prepare_wikidata_search.py ../03-json-knowledge-graph/data/level4-dataset --abbreviations ../01-abbreviation-extraction/output/merged_abbreviations.json
```

Swap `../03-json-knowledge-graph/data/level4-dataset` for
`../03-json-knowledge-graph/data/level4-dataset-full` to source candidates
from the full raw dataset instead of the small working set — expect roughly
1,000-9,000+ distinct surface forms instead of a handful, which changes the
scale of stage (b) below a lot (hundreds of SPARQL batches instead of one
or two).

For every distinct `surface_form`/`entity_type` in the extraction dataset,
builds an ordered list of `search_candidates` — the original text, an
abbreviation-expanded version (e.g. `"...sustained VA"` →
`"...sustained ventricular arrhythmia"`, expanded form always listed
*before* the bare abbreviation, since a bare abbreviation can exact-match an
unrelated real Wikidata item), and a connector-split version (`"X with Y"`
→ `"X"`, `"Y"` — a lightweight heuristic decomposition, not a full parser or
NER model). Writes `wikidata_search/wikidata_search_input.json`.

**Connector-split candidates keep the connector that produced them.**
`split_on_connectors()` used to throw the with/and/or away once it had used
it to cut the text, so `"A and B"` and `"A or B"` both collapsed into the
identical, connector-less `["A", "B"]` — nothing downstream could tell a
conjunction from an alternative apart. Each connector-split candidate now
carries `"connector"` (the word that joined it to the *previous* piece —
absent on the first piece) and `"split_index"` (its position among the
split pieces), so e.g. `"NYHA class I or II"` splitting into `"II"`
after `"NYHA class I"` is recorded with `connector: "or"`, recoverable by
any later step instead of lost at this stage.

**Every piece is now guaranteed at least one search candidate, even a bare
1-character one.** `MIN_CANDIDATE_LENGTH` (2) exists to stop a noisy short
token being searched *when a better alternative exists* — it was instead
also making a piece disappear from search entirely when it had no better
alternative. The merged abbreviation dictionary genuinely has 1-character
entries (`"R"` → `"randomized"`, `"F"` → `"feces"`, `"H"` → `"hepatic"`,
...); a short piece with no matching dictionary entry used to produce zero
candidates for its row rather than being deprioritized. `build_search_candidates()`
now falls back to the bare short text (tagged `"short_form_fallback"`, tried
last) only when nothing else qualified for that piece.

**Skips surface forms/candidates over 150 characters** (`MAX_SURFACE_FORM_LENGTH`
in `src/prepare_wikidata_search.py`) — on `level4-dataset-full`, 128 of
17,780 entities (0.7%) are whole paragraphs of clinical instructions
mis-captured as a single entity span by the stage-4 extraction (up to 752
chars), not search terms at any length; these are the same entries that
originally triggered the `HTTP 431` below. The cap applies both to the raw
surface form *and* to every candidate it produces, since abbreviation
expansion can push an under-the-limit surface form back over it. The
underlying extraction JSON files are never edited — this only affects what
becomes a Wikidata search candidate, not step 3's own graph, which still
faithfully includes these long spans as extracted. The script prints a
`Skipped N entity occurrence(s)...` line when this applies.

**Skips surface forms only ever typed `VALUE`/`DOSE`/`STRENGTH`**
(`NON_CONCEPT_ENTITY_TYPES`) — the ontology's own definitions say these are
measurements, not named concepts (e.g. `"<300 m"`, `"650 mg"`), so no
search strategy will ever match them; 799 of 8,505 surface forms on
`level4-dataset-full` (9.4%). Kept if the same text is ever also seen with
a different entity type elsewhere in the dataset.

**Adds a lowercased fallback candidate**, tried only after every
original-case candidate above — an extracted span keeps whatever
capitalization its source sentence gave it (e.g. sentence-initial
`"Haemodialysis"`), but Wikidata's canonical label for many common-noun
clinical/drug concepts is lowercase. Confirmed live: `"Haemodialysis"`
finds nothing, `"haemodialysis"` matches; same for `"Thiazolidinediones"`
vs `"thiazolidinediones"`.

Measured impact of these two changes together on `level4-dataset-full`
(300-row random sample, same methodology before/after): exact-match rate
went from **31.0% → 37.7%**. A further fuzzy-API fallback tier (like
`enrich_wikidata.py` already does for its own one-pass pipeline) was
evaluated but not adopted here yet — on a 40-row sample of the remaining
misses it only recovered 4 (10%), while costing one non-batchable HTTP
request per miss (~5,000+ requests on the full dataset) — not clearly
worth it as-is; revisit if a smarter fuzzy strategy is found.

### b. Search Wikidata and extract structured entity information

```powershell
py -3.12 src/search_wikidata_candidates.py wikidata_search/wikidata_search_input.json
py -3.12 src/expand_wikidata_entities.py wikidata_search/wikidata_matches.json
```

Run these **one at a time and wait for each to finish** rather than pasting
both (plus the rdfizer call in stage (c)) as one block. PowerShell runs
every pasted line regardless of whether an earlier one failed — if
`search_wikidata_candidates.py` dies partway through, `expand_wikidata_entities.py`
will silently run against whatever stale `wikidata_matches.json` was already
on disk (e.g. from a previous small-dataset run) instead of erroring out,
and you'll get a quietly wrong graph, not an obvious failure.

`search_wikidata_candidates.py` searches Wikidata for each row's candidates
in order, via a single batched SPARQL exact-label/alias lookup per ~50
candidate texts — not the `wbsearchentities` API or a client library, since
the actual cost is network round-trips, not per-call work. Records `P31`
(instance of) and `P921` (main subject). Writes
`wikidata_search/wikidata_matches.json`. On the full dataset this is
300-400+ batched requests (varies as candidate generation changes) against
the live Wikidata query service and takes a while (well beyond the couple
of seconds the small working set needs); it prints `Searching batch
N/TOTAL` progress as it goes. If it's interrupted or a batch's retries are
exhausted (see the checkpointing note below), just rerun the same command —
it resumes instead of starting over.

**Winning candidate/QID selection is quality-aware, not just "first thing
that matches."** Two related accuracy problems were found by inspecting
real output, both fixed in `select_match()`/`_pick_best_qid()`:

- The first candidate (in trust order: original, abbreviation-expanded,
  connector-split, lowercased) with *any* exact hit used to win outright,
  even an ambiguous, unsupported one, beating an unambiguous, well-supported
  hit on a less-trusted candidate further down the list. `select_match()`
  now only stops early on a *clean* hit — unambiguous, or ambiguous but
  resolved with a supporting signal (see below) — otherwise it keeps
  checking less-trusted candidates and falls back to the best one seen if
  none was ever clean.
- When one exact-matched text collides with more than one Wikidata item (a
  real, confirmed occurrence — e.g. "field of study" resolves to both
  `Q1047113` and `Q2267705`), the smallest QID used to win by default —
  purely a byproduct of Wikidata's own item-creation order, carrying no
  signal about which item this project's extracted `entity_type` actually
  means. Reproduced live on `level4-dataset`: the bare abbreviation `"ARB"`
  exact-matches, among others, `Q296458` ("ARB" — a Wikimedia
  disambiguation page) and `Q1026794` ("California Air Resources Board") —
  smallest-QID-wins would have picked the disambiguation page.
  `_pick_best_qid()` now excludes Wikimedia disambiguation/list/category
  pages outright (extending the existing `SCIENTIFIC_ARTICLE_TERMS`
  description filter — a paper about X is not X, and neither is a
  disambiguation page for X), then prefers whichever competing item's own
  `instance_of`/`main_subject`/description text actually supports this
  row's `entity_type`(s) (a heuristic keyword table, `TYPE_KEYWORDS` —
  grounded in data already fetched, not a second API call), then whichever
  has a description at all, and only falls back to the smallest QID as a
  last, deterministic resort — not the primary rule.

Every output row now also carries `"match_ambiguous"` — true when
`matched_text` exact-matched more than one Wikidata item and the tie had to
be resolved this way, so a downstream consumer can tell a confident
one-item match apart from a resolved-but-uncertain one (e.g. `"ARB"` above
still resolves to a non-concept organization, since no drug-class item on
Wikidata actually carries `"ARB"` as a label/alias — `match_ambiguous: true`
is exactly the signal that this particular match deserves less trust).

**Uses POST, not GET, for the batched SPARQL request** — found the hard
way: on `level4-dataset-full`, some "surface forms" are long, malformed
extraction fragments (garbled multi-clause spans), and a batch of 50 of
those embedded in a GET querystring can exceed the query service's
request-header size limit (`HTTP 431 Request Header Fields Too Large`,
reproduced live on a real batch). POST puts the query in the request body
instead, which has no comparable size ceiling.

**Retries transient failures (`429`/`5xx`/connection drops) itself, with
backoff, and paces requests proactively** — found the hard way, in two
parts:

- `enrich_wikidata.create_session()` deliberately disables retries for the
  SPARQL endpoint specifically (`max_retries=0` on that mount, separate
  from its generic-`https://` policy), so its own per-term
  exact-then-fuzzy-API search can fall back to the next tier immediately on
  any SPARQL hiccup. Correct for `enrich_wikidata.py`, but
  `search_wikidata_candidates.py` has no such fallback — a batch *is* the
  request — so reusing that session meant an ordinary rate-limit response
  killed the entire multi-hundred-batch job outright (`HTTP 429`,
  reproduced live). `run_batch()` retries `429`/`500`/`502`/`503`/`504`
  itself (up to 5 attempts, honoring `Retry-After` if sent, exponential
  backoff otherwise) — **and** separately catches connection-level failures
  with no HTTP response at all (`requests.exceptions.RequestException` —
  reproduced live as `RemoteDisconnected`, immediately after a 429, i.e.
  the query service dropped the connection outright rather than responding
  to a client it had just rate-limited). These are two different failure
  shapes needing two different `except`/`if` paths; the first version of
  this fix only handled the HTTP-status one.
- On a real full-dataset run, unpaced back-to-back requests drew a
  429/502/connection-drop about once every 12-13 batches (5 times in the
  first 63 of 373) — too often for retry-after-the-fact alone to be a good
  strategy. `search_all()` now also waits `ew.SPARQL_BATCH_PACING_SECONDS`
  (1s) between requests proactively, costing a few extra minutes across a
  full run in exchange for triggering the limit less often in the first
  place.

Both the retry/backoff and the pacing constant live in `enrich_wikidata.py`
as `sparql_request()`/`SPARQL_BATCH_PACING_SECONDS` — a shared helper, not
duplicated per script, since `expand_wikidata_entities.py`'s batch fetches
(stage (b) below) hit the exact same endpoint with the exact same "no
fallback tier, a batch IS the request" shape and needed the identical fix.

**Checkpoints after every successful batch, and resumes automatically —
found the hard way: retry/backoff isn't always enough.** A WDQS outage or
rate-limit that outlasts the retry budget (5 attempts, exponential backoff
up to 16s) still raises and stops the run — on purpose, so
`run_pipeline.py`'s `check=True` doesn't continue into `expand_wikidata_entities.py`/
step 3 against an incomplete `wikidata_matches.json` (reproduced live: a
`ReadTimeout` followed by four straight `502`s on batch 31/373). Before
this fix, that also meant losing every batch already completed — nothing
was written to disk until all N batches succeeded, so a crash 30 batches
into a 373-batch run meant redoing all 30. `search_all()` now writes
results to `<output>.checkpoint.json` after each batch; re-running the
exact same command (or `run_pipeline.py --from 2`) loads it and picks up
at the first unfinished batch instead. The checkpoint is discarded once a
run finishes all batches, and is automatically ignored (with a printed
warning) if it doesn't match the current input/`--batch-size`. Pass
`--no-resume` to force starting over from batch 1 regardless.

**`enrich_wikidata.py`'s shared `_sparql_escape()` also escapes `\n`/`\r`/`\t`,
not just `\` and `"`** — also found the hard way: on `level4-dataset-full`,
43 candidate texts contain a literal embedded newline (e.g.
`"15\nmg/kg/day"`, a PDF line-wrap artifact), spread across many different
batches. A SPARQL short string literal (`"..."`) cannot contain a raw
newline/CR/tab per the SPARQL/Turtle grammar — embedding one breaks query
parsing entirely (`HTTP 400 Bad Request`, not a graceful per-row miss),
reproduced live. Fixed at the shared escaping function both
`search_wikidata_candidates.py` and `enrich_wikidata.py`'s own single-text
search use, so it covers both.

`expand_wikidata_entities.py` expands each match into a small linked graph:
`instance_of`, `subclass_of` (P279), `main_subject`, `health_specialty`
(P1995), `symptoms` (P780), `drug_or_therapy` (P2176), plus `umls_cui`
(P2892), `drugbank_id` (P715), `also_known_as` (`skos:altLabel`), and
`matched_surface_forms` (the original searched text that reached this
match, carried over from `wikidata_matches.json` — see below) — one
hop into whatever those properties point to (a second hop gets
label/alias-only stubs, not fully expanded). For any match typed
`MEDICATION`, also reverse-searches for individual member drugs of that
class (e.g. "beta blocker" → bisoprolol, metoprolol, ...) via
`enrich_wikidata.py`'s existing `search_treatments_api`, since a drug class
itself rarely carries a DrugBank ID/UMLS CUI — only its members do. Writes
`wikidata_search/wikidata_entities.json` (a JSON array).

**IRIs are keyed by label text**, not by QID
(`http://digistrucmed.org/entity/antiarrhythmic%20agent`, not
`.../wikidata/entity/Q241549`) — a bare QID isn't meaningful on its own, and
this reuses the exact namespace step 3's `mapping-json.rml.ttl` mints
extracted surface forms under, so a Wikidata-sourced concept and a
same-text extracted entity mention land on **the same node** once both
graphs are loaded together (verified: `entity/beta%20blocker` exists in
both). Two different Wikidata items can share a label — confirmed live
(`Q1047113`/`Q2267705`, both "field of study") — when that happens both
collapse onto one node with every `wikidata_id` kept, not silently
dropped; a resulting self-referential edge is a faithful reflection of the
collision, not a bug.

**Same-text landing on the same node only works when the text actually
matches.** Stage (a)'s abbreviation-expansion and connector-splitting
routinely finds a match whose Wikidata label looks nothing like what was
searched for — e.g. `surface_form: "all individuals with HF"` matches
`wikidata_label: "heart failure"` (see `wikidata_search/wikidata_matches.json`).
Without extra bookkeeping, that connection would only exist inside
`wikidata_matches.json` — the graph would mint `entity/heart%20failure`
and never `entity/all%20individuals%20with%20HF`, leaving the two
disconnected even though the search already proved they're the same
concept. `expand_wikidata_entities.py` now carries the original
`surface_form` text forward as each hop-0 entry's `matched_surface_forms`
(deduplicated against the label the same way `also_known_as` is), and
`mapping-wikidata-entities.rml.ttl` mints each one as its own
`ds:matchedSurfaceForm` IRI edge — same IRI-per-alternate-name shape as
`ds:alias`, just recording *this project's own* search match instead of a
Wikidata-asserted synonym. Verified live: `entity/heart%20failure` now has
`ds:matchedSurfaceForm` edges to both `entity/hf` and
`entity/all%20individuals%20with%20hf`.

**Both fields are folded (casefold + collapsed whitespace) before they
become IRI text**, via `_clean_aliases`/`_normalized_key` in
`expand_wikidata_entities.py` — matching
`03-json-knowledge-graph/src/normalize_surface_forms.py`'s own fold on
every extracted `ds:SurfaceForm` node **exactly**, not just "also
lowercased." Found the hard way: an earlier version of this fix built
`ds:matchedSurfaceForm` from the *original*-cased `surface_form` text
(`entity/all%20individuals%20with%20HF`), which is a *different* IRI from
what step 3 actually mints for a matching extracted mention
(`entity/all%20individuals%20with%20hf`, after its own fold runs) — the
edge didn't error, it just pointed at an empty node with none of the real
`hasSourceMention`/`rdf:type` data on it. This RDFizer has no case-folding
function (see `normalize_surface_forms.py`'s header), so the fold has to
happen in Python before the value ever reaches `rr:template` — there's no
way to fix this inside the RML mapping itself.

**`wikidataId`/`umls_cui`/`drugbank_id` stay plain string literals**, not
IRIs like `alias`/`matchedSurfaceForm` — a QID/CUI/DrugBank ID is an
external identifier, never text a guideline would actually contain, so
there's no same-text-merge benefit to gain from minting it as an
`entity/{id}` node.

**Item-valued property values are validated as real QIDs before use** —
found the hard way: a property can be asserted with Wikidata's explicit
"unknown value" (we know it has a value, just not which one), and WDQS
returns that as a skolemized blank-node IRI
(`http://www.wikidata.org/.well-known/genid/<hash>`) rather than a real
`http://www.wikidata.org/entity/Q...` item. The old code blindly stripped
the `.../entity/` prefix off every value; on a non-item value that's a
no-op, so the full genid URL slid through untouched, got treated as a QID,
and eventually landed inside a later batch's `VALUES { wd:... }` clause as
`wd:http://www.wikidata.org/.well-known/genid/...` — invalid SPARQL syntax,
`HTTP 400 Bad Request` (not retryable, unlike `429`/`5xx`), reproduced live
242 batches into a real run, on the very last hop-2 batch. `run_entity_batch()`
now checks every extracted value against `QID_PATTERN` and skips it
otherwise — the same guard `search_wikidata_candidates.py` already had for
its own qid extraction, just missing here until now.

**Checkpoints hop 0/1/2 independently, and resumes automatically** — same
mechanism as `search_wikidata_candidates.py` (see stage (b) above), applied
here because the cost of *not* having it was worse: three sequential
batched stages (hop 0, then class-member search, then hop 1, then hop 2)
with nothing written to disk until the very end meant the `400` above, on
the last hop-2 batch, threw away 242 batches (hop 0's 61, hop 1's 104, hop
2's 76 of 77) and ~4 hours of a real run. `fetch_batched()` now writes
`<output>.hop0.checkpoint.json`/`.hop1.checkpoint.json`/`.hop2.checkpoint.json`
after every batch of its respective stage; re-running the same command
picks up at the first unfinished batch of whichever stage was in progress,
skipping every stage that already finished entirely. Class-member search
itself (the one non-batched, per-QID loop between hop 0 and hop 1) isn't
checkpointed separately — it already tolerates one QID's lookup failing on
its own (`find_class_members()`), and redoing that whole loop on a resume
triggered by a later hop 1/2 failure is cheap next to what a 100+-batch hop
redo would cost. Pass `--no-resume` to ignore all three checkpoints and
start over from hop 0.

### c. Build the RDF graph

```powershell
$env:PYTHONUTF8 = "1"
py -3.12 -m rdfizer -c config/rdfizer-wikidata-entities.ini
```

Converts `wikidata_search/wikidata_entities.json` to RDF via
`mappings/mapping-wikidata-entities.rml.ttl`, writing
`wikidata_search/knowledge_graph_wikidata_entities.nt`. Not merged into
`../03-json-knowledge-graph/output_json/knowledge_graph_json.nt`
automatically — load both `.nt` files together (e.g. in `rdflib`) to see
the connection.

## Alternative: write Wikidata matches directly into the graph

`enrich_wikidata.py` predates the 3-stage pipeline above and does a similar
job in one pass — search + property retrieval + one hop of treatment
expansion — but writes straight into a copy of step 3's RDF graph instead
of producing standalone files. Kept as-is for now; may be superseded once
the graph/Wikidata merge step above is built.

```powershell
py -3.12 enrich_wikidata.py
```

Reads `../03-json-knowledge-graph/output_json/knowledge_graph_json.nt` by
default, writes `..._wikidata.nt` + a `.report.json` next to it (input never
modified). Flags: `--source api` (skip SPARQL entirely), `--source sparql`
(no API fallback for property retrieval), `--max-related-treatments N`
(default 30), `-o FILE`. `--abbreviations FILE` is currently **stale** — it
expects an older flat-field output shape, not the current `search_candidates`
list shape stage (a) above produces.

## Tests

```powershell
py -3.12 -m pytest tests/ -q
```

`test_enrich_wikidata.py::test_navigation_keeps_two_document_grades_separate`
is a known pre-existing failure (expects `connect_graph_navigation` as an
attribute of the `enrich_wikidata` module; it only exists as the separate
script in `../03-json-knowledge-graph/src/connect_graph_navigation.py`).

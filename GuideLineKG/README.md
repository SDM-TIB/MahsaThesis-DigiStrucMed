# GuideLineKG

Builds a clinical-guideline knowledge graph under the `http://digistrucmed.org/`
namespace, in four independent, numbered steps. Each step is
self-contained — its own `data/`, `ontology/` (where relevant), `mappings/`
(RML), `config/` (RDFizer `.ini`), `src/`, and generated output — with its
own README covering exactly how to run it.

## Steps

| | Step | Depends on | What it produces |
|---|---|---|---|
| 1 | [`01-abbreviation-extraction/`](01-abbreviation-extraction/README.md) | nothing | Abbreviation glossary as RDF |
| 2 | [`02-wikidata-enrichment/`](02-wikidata-enrichment/README.md) | step 1's output + step 3's raw dataset | Structured Wikidata facts as RDF |
| 3 | [`03-json-knowledge-graph/`](03-json-knowledge-graph/README.md) | nothing (own raw dataset) | The core extraction knowledge graph, RDF |
| 4 | [`04-merge-knowledge-graphs/`](04-merge-knowledge-graphs/README.md) | steps 1-3's output | One combined, deduplicated graph |

Steps 1 and 3 have no dependency on each other and can run in either order
or in parallel. Step 2 needs step 1's `merged_abbreviations.json` and reads
(but doesn't build) step 3's raw dataset (`data/level4-dataset/` for a quick
run, `data/level4-dataset-full/` for the real one). Step 4 runs last, once
1-3 have all produced output.

Both step 2 and step 3 have a small working-set mode and a full-dataset
mode — the full one processes hundreds of files/thousands of search
candidates and takes noticeably longer (step 3: ~539 files through RDFizer
one at a time; step 2: ~250-300 batched Wikidata SPARQL requests). Each
step's own README explains the swap and what to expect. Run each pipeline
command **one at a time**, not pasted as one block — a mid-batch failure in
one command won't stop a later one from silently running against stale
leftover data from a previous run.

## Running all four steps at once

```powershell
py -3.12 run_pipeline.py
```

`run_pipeline.py` (repo root) runs steps 1-4 in order, each via a real
`subprocess.run(..., check=True)` call — not shelled-out PowerShell lines —
so a failure in any step **stops the whole run immediately** rather than
letting a later step continue against stale/partial output from a failed
one (the actual cause behind most of the "graph looks wrong" problems this
project has hit; PowerShell itself doesn't stop a pasted block just because
an earlier line failed). Prints a banner before each step and the exact
command it's running.

```powershell
py -3.12 run_pipeline.py --full       # data/level4-dataset-full instead of the
                                       # small working set - slow, hits Wikidata's
                                       # live SPARQL endpoint hundreds of times
py -3.12 run_pipeline.py --from 3     # resume from step 3 (e.g. after fixing
                                       # only step 3) instead of redoing 1-2
py -3.12 run_pipeline.py --from 2 --wikidata-from expand
                                       # step 2 only, skip straight to its
                                       # expand_wikidata_entities.py sub-stage -
                                       # e.g. after search_wikidata_candidates.py
                                       # already finished (its wikidata_matches.json
                                       # is already on disk) but a later sub-stage
                                       # crashed - see 02-wikidata-enrichment/README.md
```

Verified live end-to-end on the small working set: all four steps, ~9
minutes (step 2's live Wikidata queries dominate), same triple counts as
running each step manually.

## How the three graphs connect

All three steps mint shared/lexical concept nodes under the **same** IRI
namespace, `http://digistrucmed.org/entity/{text}` (percent-encoded surface
text) — this is the actual integration mechanism, not a metaphor. An
abbreviation (step 1), an extracted entity mention's surface form (step 3),
and a Wikidata-sourced concept (step 2) that share identical text land on
**one RDF node** once all three `.nt` outputs are loaded together, with no
`owl:sameAs`/bridging step required. Example, verified live against this
project's data:

```turtle
ds:entity/MRA a ds:Abbreviation, ds:SurfaceForm ;
    ds:hasExpansion ds:entity/Mineralocorticoid%20receptor%20antagonist .
ds:entity/beta%20blocker a ds:WikidataEntity, ds:SurfaceForm, ds:Drug ;
    ds:subclass_of ds:entity/medication ;
    ds:hasSourceMention ds:entity-mention/.../e_0012 .   # same node the JSON pipeline (step 3) mints
```

To see the combined graph, load all three outputs together, e.g. in
`rdflib`:

```python
import rdflib
g = rdflib.Graph()
g.parse("03-json-knowledge-graph/output_json/knowledge_graph_json.nt", format="nt")
g.parse("01-abbreviation-extraction/output/knowledge_graph_abbreviations.nt", format="nt")
g.parse("02-wikidata-enrichment/wikidata_search/knowledge_graph_wikidata_entities.nt", format="nt")
```

**This only merges nodes with *identical* text.** It does not fold spelling/
inflection variants - `beta-blocker` / `beta-blockers` / `beta blockers`, or
`ACEI` / `ACEIs`, stay separate nodes, each independently typed with its own
attributes, even after the three files above are loaded together. Run
[`04-merge-knowledge-graphs/`](04-merge-knowledge-graphs/README.md) on top
of the three outputs to fold those onto one canonical node per concept
(`py -3.12 04-merge-knowledge-graphs/src/merge_graphs.py`) - that's the
recommended way to get one combined, deduplicated graph, rather than the
raw three-file union above.

**Nor does it connect two *genuinely different* strings for the same
concept** - a Wikidata synonym (`ds:alias`) or a search match reached via
abbreviation-expansion/connector-splitting (`ds:matchedSurfaceForm`,
step 2's `expand_wikidata_entities.py`) is text that will never literally
match a guideline mention's node by construction. Both predicates are
IRI-valued for exactly this reason - `ds:entity/heart%20failure
ds:matchedSurfaceForm ds:entity/all%20individuals%20with%20hf` links the
two nodes explicitly, since same-text-merge alone never would (note the
lowercase `hf` - both fields are casefolded before becoming IRI text, the
same fold `normalize_surface_forms.py` applies to every extracted
`ds:SurfaceForm` node, so the two pipelines can never mint two different
IRIs for what's actually the same wording). See
[`02-wikidata-enrichment/README.md`](02-wikidata-enrichment/README.md) for
where each of these is minted.

The shared ontology (`ds:SurfaceForm`, `ds:WikidataEntity`, `ds:Drug`,
`ds:Disease`, evidence-grade classes, ...) lives at
[`03-json-knowledge-graph/ontology/ds-ontology-tables.ttl`](03-json-knowledge-graph/ontology/ds-ontology-tables.ttl) —
steps 1 and 2 extend/reuse it by IRI reference only (no file-level
dependency; there's no `owl:imports` anywhere in this repo).

## Also in this workspace

- **`kg_build/`** — a separate, older CSV/flat-SciSpacy-mentions pipeline,
  explicitly untouched by the three steps above (see `kg_build/README.md`).
  Not one of "the three steps" — it predates this reorganization and has
  its own independent inputs/outputs.

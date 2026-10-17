# Step 4 — Merge Knowledge Graphs — `04-merge-knowledge-graphs/`

Unions the three steps' `.nt` outputs into one graph and canonicalizes
surface-form nodes that are pure spelling/inflection variants of the same
concept - e.g. `beta-blocker` / `beta-blockers` / `beta blockers`, or
`ACEI` / `ACEIs` / `ACEi` / `acei`, which otherwise stay three or four
separate `ds:entity/*` nodes, each independently typed and carrying its own
`hasSourceMention`/relations, even after steps 1-3 are loaded together.

Depends on: steps 1, 2, and 3 already having been run (reads their standard
output paths by default).

## Why this step exists

Loading the three steps' `.nt` files together already merges any two nodes
that share **identical** `ds:entity/{text}` text - see the root README's
"How the three graphs connect". That's the entire cross-pipeline
integration mechanism, and it needs no code: `ds:entity/MRA` minted by step
1 and `ds:entity/MRA` extracted by step 3 are, by construction, the same
RDF node once both files are parsed into one graph.

What it does **not** do is fold spelling/inflection variants - different
text is a different IRI, full stop. Measured on this project's data: after
a plain union of the three outputs, casefold+whitespace+hyphen+plural
folding finds **~80 groups** of `ds:entity/*` nodes that are the same
concept spelled differently, each currently living as its own node with its
own attributes. This step closes that gap.

## What it merges, and what it deliberately doesn't

1. **Case + whitespace + hyphen fold (unconditional)** - `Beta-Blocker`,
   `beta  blocker`, `beta-blocker` are, without exception, the same written
   concept.
2. **Plural fold for abbreviations - via the explicit relation only.**
   `ds:hasPluralForm`/`ds:pluralFormOf` (step 1's own output) is trusted as
   the merge signal, never a blind trailing-`s` strip. This matters: `RAA`
   (right atrial appendage) and `RAAS` (renin-angiotensin-aldosterone
   *system*) merely share a text prefix - `RAAS` is not the plural of `RAA`.
   Same trap with `OSA`/`OSAS` (...*syndrome*), `SQT`/`SQTS` (...*syndrome*),
   `LQT`/`LQTS` (...*syndrome*) - all verified live in this project's data
   (`grep hasExpansion` on each pair shows genuinely different expansions).
   A step-1-extracted abbreviation only ever merges with another node via
   this relation.
3. **Plural fold for everything else - closed-world only.** For non-abbreviation
   surface forms (plain extracted clinical terms, Wikidata-sourced labels),
   merges two nodes whose folded keys differ by exactly one trailing `s`
   **only when both spellings are already present as nodes** - it never
   invents an unseen singular by stemming every node, so a genuine node like
   `status` is left alone (there is no `statu` node to collide with).
4. **Never merges qualifier/modifier phrases.** `evidence-based beta
   blockers` is a more specific concept than `beta blockers` (the qualifier
   changes meaning - a recommendation-strength distinction, not a spelling
   choice), so it's left as its own node rather than folded into the base
   concept.

See `src/merge_graphs.py`'s module docstring for the full rationale, and
`tests/test_merge_graphs.py` for the RAA/RAAS false-merge-prevention case as
an executable regression check.

## Run

From `04-merge-knowledge-graphs/`:

```powershell
py -3.12 src/merge_graphs.py
```

Reads the three steps' standard outputs:
- `../03-json-knowledge-graph/output_json/knowledge_graph_json.nt`
- `../01-abbreviation-extraction/output/knowledge_graph_abbreviations.nt`
- `../02-wikidata-enrichment/wikidata_search/knowledge_graph_wikidata_entities.nt`

Override with `--sources FILE [FILE ...]`. Writes:
- `output/knowledge_graph_merged.nt` - the combined, canonicalized graph.
- `output/merge_report.json` - one entry per merged concept, listing every
  surface-form spelling folded into it, for manual spot-checking (this is
  the artifact to scan for any new suffix-based false-merge pattern the
  RAA/RAAS-style safeguard above doesn't already cover, before trusting a
  fresh run's output).

## Cleaning up IRIs for display / CoT-training use

Two optional post-processing scripts, run after `merge_graphs.py` against
its `output/knowledge_graph_merged.nt`, that make IRIs readable without
merging or losing anything. Neither touches the canonicalization logic
above - both operate purely on the already-merged output, in place by
default. They exist because this graph's IRIs get read as text more than
resolved as URLs: any tool that verbalizes a triple by printing an IRI's
local name (a CoT/rule-mining prompt builder, a generic RDF verbalizer)
otherwise prints raw percent-encoding or an opaque 64-hex-char document
hash instead of something readable - and unlike a resolvable web IRI,
there's no requirement here that the *identifier* stay unencoded-and-ugly
just because it's also meant to be machine-addressable.

```powershell
py -3.12 src/clean_entity_iris.py
py -3.12 src/humanize_mention_iris.py
```

### `clean_entity_iris.py` — strip percent-encoding

`ds:entity/{surface_form}` IRIs (plus a handful of grade-value individuals
built the same way) are percent-encoded by the RDFizer for every character
outside the URI-unreserved set - not just spaces: parens, commas, brackets,
even multi-byte UTF-8 (`entity/hypertrophic%20cardiomyopathy`,
`entity/Hexahydrofuro%5B3%2C2-b%5Dfuran-3%2C6-diol`). Percent-decoding is
lossless (`%20` → space, `%E2%80%93` → the original character, ...), so this
just re-spells the same IRI: whitespace becomes `_` (the one substitution,
and it can't collide with anything - see the script's own docstring for
why), and everything else is percent-decoded straight back to its literal
character **except** what a strict URI validator actually rejects raw -
non-ASCII, a stray `%` not part of a valid escape, and the handful of
characters RFC 3986's path grammar reserves as structural delimiters
(`[`, `]`, `?`, `#`, plus the small IRIREF-forbidden set) - those get
re-encoded, same as before, so the file stays valid even under a strict
consumer (reproduced live: `org.eclipse.rdf4j.sail.SailException: Invalid
IRI value` on an IRI containing raw `[`/`]`). Writes
`output/clean_entity_iris_report.json` listing any two distinct original
spellings that landed on the same cleaned IRI, for review (on this
project's data: 1, and it's a false alarm - the same text with/without an
inconsistently-encoded colon).

### `humanize_mention_iris.py` — replace document hashes with readable names

`entity-mention/`, `text-unit/`, `normative-assertion/`, `grade-record/`,
and `semantic-unit/` IRIs are scoped by `lu_<64-hex-char-sha256>_n####`,
where the hash is the source document's own `source.document_sha256` -
opaque by design (it's what makes these IDs globally unique across
documents), but unreadable in a CoT prompt or a quick file scan. Every
document already has a human-readable slug sitting in the graph -
`ds:sourceDocument` literals are shaped `<slug>_<sha256[:12]>` (e.g.
`zeppenfeld-2022-esc-guidelines-va-and-scd_d1710bde778e`) - so this script
matches each hash's first 12 hex characters against that suffix and
substitutes the slug, with no external manifest needed. `entity-mention/`
nodes additionally get their own `rdfs:label` (the surface text they
point at) prepended:

```
entity-mention/lu_d1710bde...4230f712_n0019/e_0199
  -> entity-mention/lqts__zeppenfeld-2022-esc-guidelines-va-and-scd_n0019/e_0199

normative-assertion/lu_66f14f76...358e7_n0067/na_0003
  -> normative-assertion/mcdonagh-2023-esc-guidelines_n0067/na_0003
```

Can't introduce a collision: the `_n####/<xx_####>` suffix was already
globally unique on its own, and the hash→slug lookup is a fixed
one-to-one mapping, so substituting the hash for its slug (and, for
mentions, prepending a label) is purely cosmetic. Writes
`output/humanize_mention_iris_report.json` listing any node with no label
and/or no matching document slug, for review (on this project's data: 0).

## Tests

```powershell
py -3.12 -m pytest tests/ -q
```

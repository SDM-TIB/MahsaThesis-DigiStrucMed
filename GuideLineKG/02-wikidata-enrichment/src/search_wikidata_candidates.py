"""Step 3 of the Wikidata-sourcing pipeline (see ../README.md): search
Wikidata for each entity's search_candidates (from prepare_wikidata_search.py,
step 2) and record only instance of (P31) and main subject (P921) for
whatever matches - a fast, minimal first search pass. Separate from
enrich_wikidata.py (no treatment expansion, no drug property retrieval, does
not write into the knowledge graph) - that remains a distinct, older
direct-to-graph approach; merging this pipeline's output into the knowledge
graph is a planned future step, not done here.

Why batched SPARQL instead of the wbsearchentities API or a client library
(e.g. pip install Wikidata): the bottleneck for hundreds/thousands of
candidates is network round-trips, not per-call CPU work, and neither the
API loop nor a convenience wrapper library changes that - they would still
be one HTTP request per candidate. This script instead builds one SPARQL
query per BATCH of candidate texts (a VALUES clause bound to every text at
once), so a few hundred candidates cost a handful of requests instead of a
few hundred. This is only possible because the match here is exact
(rdfs:label/skos:altLabel bound directly to a literal - an indexed lookup,
no fuzzy scoring needed) - see enrich_wikidata.py's search_wikidata_exact
for the same technique used one-at-a-time in the full pipeline.

Per surface form, candidates (already ordered by prepare_wikidata_search.py:
original text, abbreviation-expanded, connector-split, ...) are tried in
order, but a hit no longer wins just by being first - see select_match()'s
docstring below. Two related accuracy problems this fixes, both found by
inspecting real output:

1. "First candidate with ANY hit wins" used to be absolute - even an
   ambiguous, unsupported hit on a trusted-but-lucky candidate would beat
   an unambiguous, well-supported hit two candidates down the list.
   select_match() now only stops early on a "clean" hit (unambiguous, or
   ambiguous but resolved with a supporting signal - see below); otherwise
   it keeps checking less-trusted candidates and can still be overridden
   by a better one, falling back to the best one seen if none was ever
   clean.
2. When one exact-matched text collides with more than one Wikidata item
   (a real, confirmed occurrence - e.g. "field of study" resolves to both
   Q1047113 and Q2267705), the smallest QID used to win by default - purely
   a byproduct of Wikidata's own item-creation order, carrying no signal
   about which item this project's extracted entity_type actually means.
   _pick_best_qid() now prefers whichever competing item's own
   instance_of/main_subject/description text actually supports this row's
   entity_type(s) (see TYPE_KEYWORDS), then whichever has a description at
   all, and only falls back to the smallest QID as a last, deterministic
   resort - not the primary rule.

Usage (run from 02-wikidata-enrichment/):
    python src/search_wikidata_candidates.py wikidata_search/wikidata_search_input.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import enrich_wikidata as ew  # noqa: E402 - reuse its session/escaping/endpoint/retry, not its full pipeline

BATCH_SIZE = 50
QID_PATTERN = re.compile(r"Q[1-9][0-9]*")

# Exact-text label/alias matches occasionally land on a Wikimedia meta-page
# rather than a real concept - a disambiguation or list page can carry the
# very same label as the concept it disambiguates, which is exactly the
# shape of collision this script has no fuzzy-scoring tier to see past.
# Checked the same way (against schema:description) as ew.SCIENTIFIC_ARTICLE_TERMS,
# whose "a paper about X is not X" reasoning applies equally here.
NON_CONCEPT_DESCRIPTION_TERMS = ew.SCIENTIFIC_ARTICLE_TERMS + (
    "wikimedia disambiguation page", "wikimedia list article",
    "wikimedia category", "wikimedia template page", "wikinews article",
)

# Keyword hints for _type_alignment_score(): weak, heuristic evidence that a
# candidate Wikidata item is the kind of thing this project's entity_type
# code means - drawn from ordinary English words a real instance_of/
# main_subject/description for that concept would plausibly use, not from
# any authoritative Wikidata taxonomy. Only used to break a tie between
# multiple QIDs that already exact-matched the same text (see
# _pick_best_qid) - never to override an unambiguous match. entity_type
# codes with no real Wikidata analogue (VALUE, DOSE, STRENGTH, TIME,
# DURATION, FORM, RATIONALE, ENTITY_OTHER - see
# ../03-json-knowledge-graph/mappings/ontology-class-lookup.json) are
# intentionally absent: they contribute no signal either way, not a penalty.
TYPE_KEYWORDS = {
    "MEDICATION": ("drug", "medication", "pharmaceutical", "chemical compound",
                    "active ingredient", "vaccine", "biologic"),
    "CONDITION_OR_FINDING": ("disease", "disorder", "syndrome", "symptom",
                              "medical sign", "medical condition", "finding", "injury"),
    "PROCEDURE_OR_INTERVENTION": ("procedure", "surgery", "therapy", "treatment",
                                   "intervention", "technique"),
    "TEST_OR_MEASUREMENT": ("test", "measurement", "assay", "diagnostic",
                             "imaging", "biomarker", "scale", "score"),
    "DEVICE": ("device", "instrument", "implant", "equipment", "apparatus"),
    "ANATOMY": ("anatomical structure", "body part", "organ", "tissue", "bone", "muscle"),
    "OUTCOME": ("outcome", "endpoint"),
    "ADVERSE_EVENT": ("adverse effect", "side effect", "adverse event", "complication", "toxicity"),
    "ROUTE": ("route of administration",),
    "FREQUENCY": ("frequency", "rate"),
    "ACTOR": ("profession", "occupation", "medical specialty", "physician"),
    "POPULATION": ("group of humans", "population", "demographic"),
}


def _type_alignment_score(match: dict, entity_types: list[str]) -> int:
    """How many of this row's entity_type(s) find a supporting keyword in
    this match's own instance_of/main_subject/description text - grounded
    in data already fetched for this match, not a second API call."""
    haystack = " ".join(match["instance_of"] + match["main_subject"] + [match["description"]]).casefold()
    return sum(1 for entity_type in entity_types
               for keyword in TYPE_KEYWORDS.get(entity_type, ())
               if keyword in haystack)


def _pick_best_qid(hits: list[dict], entity_types: list[str]) -> tuple[dict, bool]:
    """Resolve which of several same-text-matching QIDs wins. Returns
    (winning match, was_ambiguous). See the module docstring's point 2 for
    why the smallest-QID default was replaced."""
    if len(hits) == 1:
        return hits[0], False
    ranked = sorted(hits, key=lambda hit: (
        -_type_alignment_score(hit, entity_types),
        0 if hit["description"] else 1,
        int(hit["id"][1:]) if hit["id"][1:].isdigit() else 0,
    ))
    return ranked[0], True


def select_match(row: dict, matches_by_text: dict) -> tuple[dict | None, dict | None, bool]:
    """Pick the best (candidate, match, was_ambiguous) for this row across
    ALL of its search_candidates - not just the first one with any hit.
    Returns (None, None, False) if nothing matched at all.

    Stops at the first candidate whose resolved match is "clean" -
    unambiguous, or ambiguous but resolved with a positive
    _type_alignment_score - the same short-circuit the old "first hit
    wins" code had, just now gated on match quality rather than mere
    presence: an unambiguous hit still wins immediately, exactly as
    before, since that is the common, reliable case. An ambiguous hit with
    no supporting signal is provisionally kept but not trusted outright;
    scanning continues to less-trusted candidates in case one resolves
    more confidently, and the best-scoring candidate seen is used only if
    none ever turns out clean."""
    entity_types = row["entity_types"]
    best = None
    best_score = -1
    for candidate in row["search_candidates"]:
        hits = matches_by_text.get(candidate["text"])
        if not hits:
            continue
        match, ambiguous = _pick_best_qid(hits, entity_types)
        score = _type_alignment_score(match, entity_types)
        if not ambiguous or score > 0:
            return candidate, match, ambiguous
        if score > best_score:
            best, best_score = (candidate, match, ambiguous), score
    return best if best else (None, None, False)


def build_batch_query(texts):
    """One VALUES-bound exact-match lookup for every text in this batch, in
    both LANGUAGES, retrieving P31 (instance of) and P921 (main subject)."""
    literals = " ".join(
        f'"{ew._sparql_escape(text)}"@{language}'
        for text in texts for language in ew.LANGUAGES)
    return f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX schema: <http://schema.org/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?text ?item ?itemLabel ?itemDescription ?instanceOf ?instanceOfLabel ?mainSubject ?mainSubjectLabel WHERE {{
  VALUES ?text {{ {literals} }}
  {{ ?item rdfs:label ?text }} UNION {{ ?item skos:altLabel ?text }}
  OPTIONAL {{ ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel) = "en") }}
  OPTIONAL {{ ?item schema:description ?itemDescription . FILTER(LANG(?itemDescription) = "en") }}
  OPTIONAL {{ ?item wdt:P31 ?instanceOf . OPTIONAL {{ ?instanceOf rdfs:label ?instanceOfLabel . FILTER(LANG(?instanceOfLabel) = "en") }} }}
  OPTIONAL {{ ?item wdt:P921 ?mainSubject . OPTIONAL {{ ?mainSubject rdfs:label ?mainSubjectLabel . FILTER(LANG(?mainSubjectLabel) = "en") }} }}
}}
"""


def run_batch(session, texts):
    """Return {text: [{"id", "label", "description", "instance_of": [...],
    "main_subject": [...]}, ...]} for this batch of texts, best match first.

    POST, not GET: a batch's VALUES clause embeds every candidate text twice
    (once per LANGUAGE) - on the full level4-dataset-full corpus, some
    candidate "texts" are long, malformed extraction fragments (garbled
    multi-clause spans), and a batch of 50 of those as a GET querystring can
    exceed the query service's request-header size limit (HTTP 431 - seen
    live on batch 4/293 of a real run). POST puts the query in the request
    body instead, which has no comparable size ceiling.

    Retry/backoff itself is ew.sparql_request() - see its docstring and
    create_session()'s for why this script needs its own retries rather
    than relying on the shared session's policy."""
    response = ew.sparql_request(session, "post", data={
        "query": build_batch_query(texts), "format": "json",
    })

    accumulated = {}  # (text, qid) -> accumulator
    for row in response.json()["results"]["bindings"]:
        text = row["text"]["value"]
        qid = row["item"]["value"].removeprefix("http://www.wikidata.org/entity/")
        if not QID_PATTERN.fullmatch(qid):
            continue  # Properties (P...) and other non-item IRIs can also carry rdfs:label/skos:altLabel.
        entry = accumulated.setdefault((text, qid), {
            "id": qid, "label": row.get("itemLabel", {}).get("value", ""),
            "description": row.get("itemDescription", {}).get("value", ""),
            "instance_of": set(), "main_subject": set(),
        })
        if "instanceOf" in row:
            label = row.get("instanceOfLabel", {}).get("value", "")
            code = row["instanceOf"]["value"].removeprefix("http://www.wikidata.org/entity/")
            entry["instance_of"].add(f"{code}: {label}" if label else code)
        if "mainSubject" in row:
            label = row.get("mainSubjectLabel", {}).get("value", "")
            code = row["mainSubject"]["value"].removeprefix("http://www.wikidata.org/entity/")
            entry["main_subject"].add(f"{code}: {label}" if label else code)

    by_text = {}
    for (text, qid), entry in accumulated.items():
        if any(term in entry["description"].casefold() for term in NON_CONCEPT_DESCRIPTION_TERMS):
            continue
        entry["instance_of"] = sorted(entry["instance_of"])
        entry["main_subject"] = sorted(entry["main_subject"])
        by_text.setdefault(text, []).append(entry)
    for matches in by_text.values():
        # A merely deterministic base ordering, not the tie-break itself
        # anymore - select_match()/_pick_best_qid() decide the real winner
        # when a text has more than one entry here; QID order is only what
        # they fall back to once every other signal is exhausted.
        matches.sort(key=lambda m: int(m["id"][1:]) if m["id"][1:].isdigit() else 0)
    return by_text


def _load_checkpoint(path: Path, total_texts: int, batch_size: int) -> tuple[dict, int]:
    """(results_so_far, first_batch_index_still_to_run) - (., 1) if no
    checkpoint exists, or if one exists but doesn't match this run's input/
    batch-size (e.g. a different --input) closely enough to trust."""
    if not path.exists():
        return {}, 1
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}, 1
    if data.get("batch_size") != batch_size or data.get("total_texts") != total_texts:
        print(f"  Ignoring stale checkpoint at {path} (input or --batch-size changed)", flush=True)
        return {}, 1
    completed = data.get("completed_batches", 0)
    print(f"  Resuming from checkpoint: {completed} batch(es) already done -> {path}", flush=True)
    return data.get("results", {}), completed + 1


def _save_checkpoint(path: Path, results: dict, completed_batches: int, total_texts: int, batch_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "batch_size": batch_size, "total_texts": total_texts,
        "completed_batches": completed_batches, "results": results,
    }, ensure_ascii=False), encoding="utf-8")


def search_all(session, texts, batch_size=BATCH_SIZE, checkpoint_path: Path | None = None):
    """Runs every batch in sequence, with a fixed pause between requests.

    The pacing is proactive, not just run_batch()'s reactive retry-on-429:
    on a real level4-dataset-full run, back-to-back unpaced requests drew a
    429/502/connection-drop about once every 12-13 batches (5 times in the
    first 63 of 373) - too often to be occasional bad luck, and each retry
    without slowing the overall rate down just walks back into the same
    limit. A flat pause between requests (ew.SPARQL_BATCH_PACING_SECONDS)
    costs a few extra minutes across a full run but should avoid triggering
    the limit in the first place, rather than only recovering from it after
    the fact.

    When run_batch() eventually exhausts ITS OWN retries (a WDQS outage
    lasting longer than ~30s of backoff, not just a blip) it raises, and
    that propagates out of here uncaught - by design, so run_pipeline.py's
    check=True still stops the whole pipeline rather than continuing on
    incomplete data. What's different from before is that every batch's
    result is checkpointed to disk as soon as it succeeds, so that crash
    only ever costs the ONE batch in flight - rerunning the exact same
    command (or `run_pipeline.py --from 2`) picks back up at that batch
    instead of redoing all the ones before it."""
    texts = sorted(set(texts))
    total_batches = -(-len(texts) // batch_size)
    results, start_index = ({}, 1) if checkpoint_path is None else _load_checkpoint(
        checkpoint_path, len(texts), batch_size)

    for index, start in enumerate(range(0, len(texts), batch_size), 1):
        if index < start_index:
            continue
        batch = texts[start:start + batch_size]
        print(f"Searching batch {index}/{total_batches} ({len(batch)} text(s))...", flush=True)
        if index > start_index:
            time.sleep(ew.SPARQL_BATCH_PACING_SECONDS)
        results.update(run_batch(session, batch))
        if checkpoint_path is not None:
            _save_checkpoint(checkpoint_path, results, index, len(texts), batch_size)

    if checkpoint_path is not None:
        checkpoint_path.unlink(missing_ok=True)  # job finished - stop offering a stale resume next time
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="prepare_wikidata_search.py output")
    parser.add_argument("-o", "--output", type=Path, default=Path("wikidata_search/wikidata_matches.json"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                         help="Distinct texts per SPARQL query (default: %(default)s)")
    parser.add_argument("--no-resume", action="store_true",
                         help="Ignore any existing checkpoint and start this run from batch 1")
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    all_texts = [c["text"] for row in rows for c in row["search_candidates"]]

    checkpoint_path = args.output.with_name(args.output.name + ".checkpoint.json")
    if args.no_resume:
        checkpoint_path.unlink(missing_ok=True)

    with ew.create_session() as session:
        matches_by_text = search_all(session, all_texts, args.batch_size, checkpoint_path)

    output_rows = []
    matched = 0
    ambiguous_matches = 0
    for row in rows:
        matched_candidate, match, ambiguous = select_match(row, matches_by_text)
        if match:
            matched += 1
            ambiguous_matches += int(ambiguous)
        output_rows.append({
            "surface_form": row["surface_form"],
            "entity_types": row["entity_types"],
            "matched_text": matched_candidate["text"] if matched_candidate else None,
            "matched_derivation": matched_candidate["derivation"] if matched_candidate else None,
            "wikidata_id": match["id"] if match else None,
            "wikidata_label": match["label"] if match else None,
            "instance_of": match["instance_of"] if match else [],
            "main_subject": match["main_subject"] if match else [],
            # True when matched_text exact-matched more than one Wikidata
            # item and _pick_best_qid() had to resolve the tie (see its
            # docstring) - a flag for how much to trust this particular
            # match, not an error.
            "match_ambiguous": bool(match) and ambiguous,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(output_rows)} surface form(s), {matched} matched "
          f"({ambiguous_matches} resolved from an ambiguous exact-text match) -> {args.output}")


if __name__ == "__main__":
    main()

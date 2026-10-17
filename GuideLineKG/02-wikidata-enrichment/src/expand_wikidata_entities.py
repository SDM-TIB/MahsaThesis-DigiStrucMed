"""Step 3b of the Wikidata-sourcing pipeline (see ../README.md): expand
each Wikidata match from search_wikidata_candidates.py (step 3a) into a
small linked knowledge graph: the matched item's own properties, plus one
hop of expansion into whatever those properties point to - not further than
that hop, by explicit choice.

Nodes are keyed by LABEL TEXT, not by QID - each entry's "label" field is
its identity, and every relationship field (instance_of, subclass_of, ...)
is an array of LABEL STRINGS referencing another entry's "label" in this
same file, not a QID and not a pre-built IRI. A bare QID is not meaningful
on its own to a human or an LLM reading this data, and this file is not
meant to stay a live pointer into Wikidata - it exists to pull facts OUT of
Wikidata into locally meaningful, connectable records. Two different
Wikidata items CAN share an English label (a real risk this design
accepts) - when that happens here, they collapse onto one record and their
QIDs are both kept in "wikidata_ids" so nothing is silently lost, just not
used as the primary key.

Each entry also carries a convenience "iri" field
(http://digistrucmed.org/entity/antiarrhythmic%20agent) for whenever this
is turned into RDF later - not used for the relationship arrays themselves.
Building the knowledge graph from this file is intentionally a separate,
later step (not done by this script) - see ../README.md.

Three "hop" levels, tagged per entry as the MINIMUM hop any merged QID was
reached at:
  0 - a matched item itself (from wikidata_matches.json) - full properties.
  1 - an item some hop-0 entity's property points to - ALSO full properties.
      Reached two ways, both recorded in "discovered_via":
        - "forward_property": found by following a hop-0 item's own P31/
          P279/P921/P1995/P780/P2176 values forward.
        - "class_member_search": for a hop-0 item with entity_type
          MEDICATION, a reverse lookup for items that have THIS item as
          their own P31/P279/P2868 (class members) and carry a DrugBank ID
          (P715) - a drug class like "beta blocker" has no forward property
          pointing down to its member drugs (bisoprolol, metoprolol, ...),
          those only carry DrugBank IDs/UMLS CUIs on themselves. Same
          technique enrich_wikidata.py's
          search_treatments_api(class_members=True) already uses.
  2 - an item some hop-1 entity's property points to - label/alias/
      description only ("stub": true, no relationship arrays fetched) - not
      further expanded. If a hop-2 reference resolves to the SAME label as
      a hop-0/1 entry, it is merged into that entry rather than kept as a
      separate stub (same IRI either way, since IRIs are label-keyed).

Properties fetched (real Wikidata PIDs):
    instance_of        P31   (item-valued, expanded)
    subclass_of         P279  (item-valued, expanded)
    main_subject        P921  (item-valued, expanded)
    health_specialty     P1995 (item-valued, expanded)
    symptoms            P780  (item-valued, expanded)
    drug_or_therapy      P2176 (item-valued, expanded)
    umls_cuis           P2892 (string, repeatable - literal fact(s))
    drugbank_ids         P715  (string, repeatable - literal fact(s))
    also_known_as        skos:altLabel (string, repeatable - "Also known as")

Each hop-0 entry also carries "matched_surface_forms": the original
wikidata_matches.json "surface_form" text(s) that search_wikidata_candidates.py
matched to reach this item - NOT the same as also_known_as (a Wikidata-
sourced fact) or "label" (this entry's own identity). A match found via
abbreviation-expansion or connector-splitting means this text can differ
from the label - e.g. "all individuals with HF" matching Wikidata's "heart
failure" - and without recording it here, that connection is lost:
nothing downstream could tell a step-3-extracted mention of "all
individuals with HF" and this Wikidata concept were ever related. This is
the field mapping-wikidata-entities.rml.ttl mints ds:matchedSurfaceForm
IRIs from.

"Possible medical findings" was intentionally left out - no single
Wikidata property clearly corresponds to it.

Output is a JSON ARRAY (not a dict), one object per merged label.

Usage (run from 02-wikidata-enrichment/):
    python src/expand_wikidata_entities.py wikidata_search/wikidata_matches.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import enrich_wikidata as ew  # noqa: E402 - reuse its session/endpoint/retry, not its full pipeline

ENTITY_PREFIX = "http://digistrucmed.org/entity/"
BATCH_SIZE = 25
QID_PATTERN = re.compile(r"Q[1-9][0-9]*")

ITEM_PROPERTIES = {
    "instance_of": "P31",
    "subclass_of": "P279",
    "main_subject": "P921",
    "health_specialty": "P1995",
    "symptoms": "P780",
    "drug_or_therapy": "P2176",
}
STRING_PROPERTIES = {
    "umls_cui": "P2892",
    "drugbank_id": "P715",
}
ALL_PROPERTIES = {**ITEM_PROPERTIES, **STRING_PROPERTIES}
PID_TO_NAME = {pid: name for name, pid in ALL_PROPERTIES.items()}


def label_iri(text: str) -> str:
    return ENTITY_PREFIX + quote(text, safe="")


def build_entity_query(qids: list[str]) -> str:
    """Full properties for each of these items: label, aliases, description,
    every ITEM_PROPERTIES/STRING_PROPERTIES value. Relationship targets are
    resolved to a label via the global qid_to_label map built from every
    tier's own fetch, not a second label column here - every target QID is
    itself fetched at some hop (1 or 2), so its own "itemLabel" is always
    available by the time relationships are resolved in main()."""
    values = " ".join(f"wd:{qid}" for qid in qids)
    prop_values = " ".join(f"wdt:{pid}" for pid in ALL_PROPERTIES.values())
    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX schema: <http://schema.org/>
SELECT ?item ?itemLabel ?itemDescription ?altLabel ?p ?value WHERE {{
  VALUES ?item {{ {values} }}
  OPTIONAL {{ ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel) = "en") }}
  OPTIONAL {{ ?item schema:description ?itemDescription . FILTER(LANG(?itemDescription) = "en") }}
  OPTIONAL {{ ?item skos:altLabel ?altLabel . FILTER(LANG(?altLabel) IN ("en", "de")) }}
  OPTIONAL {{
    VALUES ?p {{ {prop_values} }}
    ?item ?p ?value .
  }}
}}
"""


def build_label_query(qids: list[str]) -> str:
    """Identity only - label, aliases, description, no property fan-out.
    Used for hop-2 stubs, which are referenced but not themselves expanded."""
    values = " ".join(f"wd:{qid}" for qid in qids)
    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX schema: <http://schema.org/>
SELECT ?item ?itemLabel ?itemDescription ?altLabel WHERE {{
  VALUES ?item {{ {values} }}
  OPTIONAL {{ ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel) = "en") }}
  OPTIONAL {{ ?item schema:description ?itemDescription . FILTER(LANG(?itemDescription) = "en") }}
  OPTIONAL {{ ?item skos:altLabel ?altLabel . FILTER(LANG(?altLabel) IN ("en", "de")) }}
}}
"""


def _new_entity(qid: str) -> dict:
    return {"label": "", "description": "", "also_known_as": set(), "targets": set()}


def run_entity_batch(session, qids: list[str]) -> dict[str, dict]:
    # ew.sparql_request retries transient failures itself (429/5xx and
    # connection drops) - this script has no fallback tier to jump to on a
    # batch failure, same reasoning as search_wikidata_candidates.py's
    # run_batch(); see create_session()'s docstring for why the shared
    # session alone won't retry SPARQL calls.
    response = ew.sparql_request(session, "get", params={
        "query": build_entity_query(qids), "format": "json",
    })

    entities = {qid: _new_entity(qid) for qid in qids}
    for row in response.json()["results"]["bindings"]:
        qid = row["item"]["value"].removeprefix("http://www.wikidata.org/entity/")
        entity = entities[qid]
        if row.get("itemLabel", {}).get("value"):
            entity["label"] = row["itemLabel"]["value"]
        if row.get("itemDescription", {}).get("value"):
            entity["description"] = row["itemDescription"]["value"]
        if row.get("altLabel", {}).get("value"):
            entity["also_known_as"].add(row["altLabel"]["value"])
        if "p" not in row:
            continue
        pid = row["p"]["value"].removeprefix("http://www.wikidata.org/prop/direct/")
        name = PID_TO_NAME.get(pid)
        if not name:
            continue
        if name in STRING_PROPERTIES:
            entity.setdefault(name, set()).add(row["value"]["value"])
        else:
            target_qid = row["value"]["value"].removeprefix("http://www.wikidata.org/entity/")
            if not QID_PATTERN.fullmatch(target_qid):
                # An item-valued property can be asserted as Wikidata's
                # explicit "unknown value" (we know it has a value, just not
                # which one) - WDQS returns that as a skolemized blank-node
                # IRI (http://www.wikidata.org/.well-known/genid/<hash>), not
                # a real Q-item. removeprefix() is a no-op on that (wrong
                # prefix), so the full URL used to slide through untouched,
                # get treated as a QID, and eventually poison a later batch's
                # `VALUES { wd:... }` clause with `wd:http://...` - invalid
                # SPARQL syntax, HTTP 400, reproduced live on hop-2 batch
                # 77/77 of a real run (search_wikidata_candidates.py already
                # guards its own qid extraction the same way; this one
                # hadn't).
                continue
            entity["targets"].add(target_qid)
            entity.setdefault(name, set()).add(target_qid)
    return entities


def run_label_batch(session, qids: list[str]) -> dict[str, dict]:
    response = ew.sparql_request(session, "get", params={
        "query": build_label_query(qids), "format": "json",
    })
    entities = {qid: _new_entity(qid) for qid in qids}
    for row in response.json()["results"]["bindings"]:
        qid = row["item"]["value"].removeprefix("http://www.wikidata.org/entity/")
        entity = entities[qid]
        if row.get("itemLabel", {}).get("value"):
            entity["label"] = row["itemLabel"]["value"]
        if row.get("itemDescription", {}).get("value"):
            entity["description"] = row["itemDescription"]["value"]
        if row.get("altLabel", {}).get("value"):
            entity["also_known_as"].add(row["altLabel"]["value"])
    return entities


def _entity_to_jsonable(entity: dict) -> dict:
    return {key: sorted(value) if isinstance(value, set) else value for key, value in entity.items()}


def _entity_from_jsonable(data: dict) -> dict:
    return {key: set(value) if isinstance(value, list) else value for key, value in data.items()}


def _load_fetch_checkpoint(path: Path, total_items: int, batch_size: int) -> tuple[dict, int]:
    """(results_so_far, first_batch_index_still_to_run) - (., 1) if no
    checkpoint exists or it doesn't match this run's qid count/batch-size."""
    if not path.exists():
        return {}, 1
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}, 1
    if data.get("batch_size") != batch_size or data.get("total_items") != total_items:
        print(f"  Ignoring stale checkpoint at {path} (input or --batch-size changed)", flush=True)
        return {}, 1
    completed = data.get("completed_batches", 0)
    print(f"  Resuming from checkpoint: {completed} batch(es) already done -> {path}", flush=True)
    results = {qid: _entity_from_jsonable(entity) for qid, entity in data.get("results", {}).items()}
    return results, completed + 1


def _save_fetch_checkpoint(path: Path, results: dict, completed_batches: int, total_items: int, batch_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    jsonable_results = {qid: _entity_to_jsonable(entity) for qid, entity in results.items()}
    path.write_text(json.dumps({
        "batch_size": batch_size, "total_items": total_items,
        "completed_batches": completed_batches, "results": jsonable_results,
    }, ensure_ascii=False), encoding="utf-8")


def _run_batch_resilient(session, run_batch, batch: list[str]) -> dict:
    """run_batch already gets ew.sparql_request's own retry/backoff for
    transient failures (429/5xx, connection drops, malformed JSON bodies).
    This handles what's left: a failure that repeats identically on every
    retry - reproduced live as the exact same JSONDecodeError byte offset
    five times in a row on one 25-item batch - which means retrying the
    SAME batch again is pointless; something about (at least one item in)
    THIS batch is what WDQS can't serialize, not the network. Bisects the
    batch and retries each half separately, isolating whichever QID(s) are
    actually responsible instead of losing full property data for 25 items
    over one bad one. A batch of one that still fails is given up on (a
    label/alias/description-only stub, same shape as a hop-2 entry) - same
    per-item tolerance find_class_members() already applies to its own
    lookups, since the rest of the run matters more than one item's
    relationships."""
    try:
        return run_batch(session, batch)
    except (RuntimeError, ew.requests.RequestException) as error:
        if len(batch) == 1:
            print(f"  giving up on {batch[0]}: {error}", flush=True)
            return {batch[0]: _new_entity(batch[0])}
        mid = len(batch) // 2
        print(f"  batch of {len(batch)} failed ({error}); splitting into "
              f"{mid} + {len(batch) - mid} and retrying separately", flush=True)
        left = _run_batch_resilient(session, run_batch, batch[:mid])
        time.sleep(ew.SPARQL_BATCH_PACING_SECONDS)
        right = _run_batch_resilient(session, run_batch, batch[mid:])
        return {**left, **right}


def fetch_batched(session, qids: list[str], batch_size: int, run_batch,
                   checkpoint_path: Path | None = None) -> dict:
    # Paced proactively, same reasoning as search_wikidata_candidates.py's
    # search_all(): unpaced back-to-back SPARQL requests draw rate-limit/
    # connection-drop responses often enough on a real run to be worth
    # avoiding up front, not just recovering from after the fact.
    #
    # Checkpoints after every batch, same mechanism/reasoning as
    # search_wikidata_candidates.py's search_all() (see its own docstring
    # and the root README) - found necessary the same way: a run that got
    # through hop 0 (61 batches), class-member search, and hop 1 (104
    # batches) crashed on the very last hop-2 batch (77/77) after ~4 hours,
    # and without this, that meant redoing all 242 batches, not just the
    # one that failed. Each of hop 0/1/2's fetch_batched() call gets its
    # own checkpoint file (different qid sets, different points in main()).
    # Class-member search (the one non-batched, per-QID loop between hop 0
    # and hop 1) isn't checkpointed separately - find_class_members()
    # already tolerates a single QID's lookup failing on its own, and
    # redoing that whole loop on a resume is cheap next to what a 100+
    # batch hop 1/2 redo would cost.
    qids = sorted(set(qids))
    total = -(-len(qids) // batch_size) if qids else 0
    results, start_index = ({}, 1) if checkpoint_path is None else _load_fetch_checkpoint(
        checkpoint_path, len(qids), batch_size)

    for index, start in enumerate(range(0, len(qids), batch_size), 1):
        if index < start_index:
            continue
        batch = qids[start:start + batch_size]
        print(f"  batch {index}/{total} ({len(batch)} item(s))...", flush=True)
        if index > start_index:
            time.sleep(ew.SPARQL_BATCH_PACING_SECONDS)
        results.update(_run_batch_resilient(session, run_batch, batch))
        if checkpoint_path is not None:
            _save_fetch_checkpoint(checkpoint_path, results, index, len(qids), batch_size)

    if checkpoint_path is not None:
        checkpoint_path.unlink(missing_ok=True)  # this stage finished - stop offering a stale resume next time
    return results


def find_class_members(session, qid: str, limit: int) -> list[str]:
    """Reverse lookup: items with qid as their own P31/P279/P2868 AND a
    DrugBank ID - the individual drugs inside a drug class. Wraps
    enrich_wikidata.py's search_treatments_api instead of reimplementing
    the haswbstatement: search-API query and pagination."""
    try:
        member_ids, _truncated = ew.search_treatments_api(session, qid, limit, class_members=True)
        return member_ids
    except (ew.requests.RequestException, ValueError, KeyError) as error:
        print(f"  class-member search failed for {qid}: {error}", flush=True)
        return []


def resolve_label(qid: str, qid_to_label: dict[str, str]) -> str:
    """Fall back to the bare QID if this item genuinely has no English
    label, so a relationship IRI is never built from an empty string."""
    return qid_to_label.get(qid) or qid


def merge_into_groups(groups: dict[str, dict], qid_to_label: dict[str, str],
                       qids: list[str], source: dict[str, dict], hop: int,
                       discovered_via: dict[str, set[str]], has_data: bool,
                       surface_forms: dict[str, set[str]] | None = None) -> None:
    """Fold each qid's fetched entity into the group for its resolved
    label, creating the group on first use. has_data=False marks a hop-2
    (label-only) fetch - contributes identity info but no relationships,
    and does not by itself make the merged group non-stub.

    surface_forms (only ever passed for the hop-0/root call - hop 1/2 items
    were never themselves searched for, they were reached by property
    traversal) is {qid: {the original wikidata_matches.json "surface_form"
    text(s) that searched-and-matched this qid}}. Without this, a match
    found via abbreviation-expansion or connector-splitting (e.g. "all
    individuals with HF" matching Wikidata's "heart failure") is invisible
    outside this script - main()'s root_qids/hop0 fetch only ever carries
    the matched QID forward, never the original search text, so nothing
    downstream could previously tell a step-3-extracted mention like
    "all individuals with HF" and this Wikidata concept were the same
    thing. See ds:matchedSurfaceForm in finalize_group below."""
    for qid in qids:
        entity = source[qid]
        label = resolve_label(qid, qid_to_label)
        group = groups.setdefault(label, {
            "label": label, "description": "", "also_known_as": set(),
            "wikidata_ids": set(), "hop": hop, "has_data": False,
            "discovered_via": set(), "matched_surface_forms": set(),
            **{name: set() for name in ITEM_PROPERTIES},
            **{name + "s": set() for name in STRING_PROPERTIES},
        })
        group["wikidata_ids"].add(qid)
        group["also_known_as"] |= entity["also_known_as"]
        group["matched_surface_forms"] |= (surface_forms or {}).get(qid, set())
        if not group["description"] and entity["description"]:
            group["description"] = entity["description"]
        group["hop"] = min(group["hop"], hop)
        default_via = {"match"} if hop == 0 else {"reference"} if hop == 2 else set()
        group["discovered_via"] |= discovered_via.get(qid, default_via)
        if has_data:
            group["has_data"] = True
            for name in STRING_PROPERTIES:
                group[name + "s"] |= entity.get(name, set())
            for name in ITEM_PROPERTIES:
                for target_qid in entity.get(name, set()):
                    # Plain label text, not a pre-built IRI: every other
                    # cross-reference in this file (also_known_as lookups,
                    # the eventual RML mapping) resolves a node by its
                    # "label" field via the same rr:template Wikidata
                    # subject IRIs already use - baking a second,
                    # independently-escaped IRI string in here would only
                    # need to be unescaped again downstream, and a stray
                    # mismatch between the two escaping paths is exactly
                    # what produced a broken (double percent-encoded)
                    # relationship IRI the one time this was tried directly.
                    group[name].add(resolve_label(target_qid, qid_to_label))


def _normalized_key(text: str) -> str:
    """Same fold ../../03-json-knowledge-graph/src/normalize_surface_forms.py's
    _normalized_key uses on every extracted ds:SurfaceForm node (casefold +
    collapsed whitespace). also_known_as/matched_surface_forms values become
    ds:alias/ds:matchedSurfaceForm IRIs via rr:template, which can't casefold
    itself - there is no such function in this RDFizer (see that script's own
    header comment) - so the VALUE must already be pre-folded here, or the
    resulting IRI silently lands on a different node than the one step 3's
    own normalization converges a matching extracted mention to. Reproduced
    live: a ds:matchedSurfaceForm edge built from the raw text
    "all individuals with HF" pointed at entity/all%20individuals%20with%20HF,
    a bare node with zero other triples - the real extracted node, with
    hasSourceMention/rdf:type/everything else, was at
    entity/all%20individuals%20with%20hf after step 3's own fold ran. The
    edge didn't error, it just didn't connect to anything."""
    return " ".join(text.casefold().split())


def _clean_aliases(aliases: set[str], label: str) -> list[str]:
    """Wikidata's en/de skos:altLabel query (build_entity_query/build_label_query
    above) routinely returns the same alias twice differing only by case - e.g.
    dyclonine's "1-(4-Butoxyphenyl)-3-(1-piperidinyl)-1-propanone" (en) and
    "1-(4-butoxyphenyl)-3-(1-piperidinyl)-1-propanone" (de) - or an alias that
    is just the canonical label in different casing ("Dyclonine" vs label
    "dyclonine"). The exact-string set this is built from (also_known_as
    above) doesn't catch either, since a Python set only drops exact
    duplicates. Keep one folded form per distinct alias, and drop any alias
    that's the label in another casing - both would otherwise double
    `ds:alias` triples on the same entity for no informational gain (18
    aliases for dyclonine before this, 9 after, one real chemical synonym
    each). The output is the folded (_normalized_key) form, not the
    original casing - it has to be, so the IRI rr:template builds from it
    is the same IRI step 3's own normalization would converge a matching
    extracted mention to (see _normalized_key's docstring); the original
    casing isn't lost, it just isn't what this particular bridge-building
    field is for - rdfs:label/ds:name on the actual extracted node keep it."""
    label_key = _normalized_key(label)
    keys = set()
    for value in aliases:
        key = _normalized_key(value)
        if key != label_key:
            keys.add(key)
    return sorted(keys)


def finalize_group(label: str, group: dict) -> dict:
    entry = {
        "iri": label_iri(label), "label": label, "description": group["description"],
        "also_known_as": _clean_aliases(group["also_known_as"], label),
        # Reuses _clean_aliases' case-insensitive dedup / drop-if-equals-label
        # logic - same shape of problem (a set of alternate strings for one
        # concept), just sourced from this project's own search matches
        # instead of Wikidata's skos:altLabel.
        "matched_surface_forms": _clean_aliases(group["matched_surface_forms"], label),
        "wikidata_ids": sorted(group["wikidata_ids"]),
        "hop": group["hop"], "stub": not group["has_data"],
        "discovered_via": sorted(group["discovered_via"]),
    }
    for name in STRING_PROPERTIES:
        entry[name + "s"] = sorted(group[name + "s"])
    for name in ITEM_PROPERTIES:
        entry[name] = sorted(group[name])
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="search_wikidata_candidates.py output (wikidata_matches.json)")
    parser.add_argument("-o", "--output", type=Path, default=Path("wikidata_search/wikidata_entities.json"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                         help="QIDs per SPARQL query (default: %(default)s)")
    parser.add_argument("--class-member-limit", type=int, default=30,
                         help="Max member drugs discovered per MEDICATION-typed match (default: %(default)s)")
    parser.add_argument("--no-resume", action="store_true",
                         help="Ignore any existing hop 0/1/2 checkpoints and start this run from scratch")
    args = parser.parse_args()

    hop0_checkpoint = args.output.with_name(args.output.name + ".hop0.checkpoint.json")
    hop1_checkpoint = args.output.with_name(args.output.name + ".hop1.checkpoint.json")
    hop2_checkpoint = args.output.with_name(args.output.name + ".hop2.checkpoint.json")
    if args.no_resume:
        for path in (hop0_checkpoint, hop1_checkpoint, hop2_checkpoint):
            path.unlink(missing_ok=True)

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    root_qids = sorted({row["wikidata_id"] for row in rows if row.get("wikidata_id")})
    medication_qids = sorted({row["wikidata_id"] for row in rows
                               if row.get("wikidata_id") and "MEDICATION" in row.get("entity_types", [])})
    # {qid: {surface_form text(s) search_wikidata_candidates.py matched to
    # it}} - the only place this file's "surface_form" field is read at
    # all, otherwise silently dropped (see merge_into_groups' docstring).
    qid_to_surface_forms: dict[str, set[str]] = {}
    for row in rows:
        if row.get("wikidata_id") and row.get("surface_form"):
            qid_to_surface_forms.setdefault(row["wikidata_id"], set()).add(row["surface_form"])

    with ew.create_session() as session:
        print(f"Hop 0: fetching {len(root_qids)} matched item(s)...", flush=True)
        hop0 = fetch_batched(session, root_qids, args.batch_size, run_entity_batch, hop0_checkpoint)

        discovered_via: dict[str, set[str]] = {}
        forward_qids = {qid for entity in hop0.values() for qid in entity["targets"]} - set(root_qids)
        for qid in forward_qids:
            discovered_via.setdefault(qid, set()).add("forward_property")

        print(f"Class-member search: {len(medication_qids)} MEDICATION-typed match(es)...", flush=True)
        for qid in medication_qids:
            members = find_class_members(session, qid, args.class_member_limit)
            if members:
                print(f"  {qid} ({hop0[qid]['label']}): {len(members)} member drug(s)", flush=True)
            for member_qid in members:
                if member_qid not in root_qids:
                    discovered_via.setdefault(member_qid, set()).add("class_member_search")

        hop1_qids = sorted(discovered_via)
        print(f"Hop 1: fetching {len(hop1_qids)} linked item(s)...", flush=True)
        hop1 = fetch_batched(session, hop1_qids, args.batch_size, run_entity_batch, hop1_checkpoint)

        known = set(root_qids) | set(hop1_qids)
        hop2_qids = sorted({qid for entity in hop1.values() for qid in entity["targets"]} - known)
        print(f"Hop 2 (labels only): fetching {len(hop2_qids)} referenced item(s)...", flush=True)
        hop2 = fetch_batched(session, hop2_qids, args.batch_size, run_label_batch, hop2_checkpoint)

    qid_to_label = {qid: entity["label"] for source in (hop0, hop1, hop2) for qid, entity in source.items()}

    groups: dict[str, dict] = {}
    merge_into_groups(groups, qid_to_label, root_qids, hop0, 0, discovered_via, has_data=True,
                       surface_forms=qid_to_surface_forms)
    merge_into_groups(groups, qid_to_label, hop1_qids, hop1, 1, discovered_via, has_data=True)
    merge_into_groups(groups, qid_to_label, hop2_qids, hop2, 2, discovered_via, has_data=False)

    output = [finalize_group(label, group) for label, group in sorted(groups.items())]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    full = sum(1 for e in output if not e["stub"])
    class_member_count = sum(1 for v in discovered_via.values() if "class_member_search" in v)
    collisions = sum(1 for e in output if len(e["wikidata_ids"]) > 1)
    print(f"{len(root_qids)} matched + {len(hop1_qids)} hop-1 ({class_member_count} via class-member search) "
          f"+ {len(hop2_qids)} hop-2 stub, merged by label into {len(output)} node(s) "
          f"({full} with data, {len(output) - full} stub, {collisions} label collision(s)) -> {args.output}")


if __name__ == "__main__":
    main()

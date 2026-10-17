"""Read this file from main() at the bottom: it runs these five steps.

1. Read the existing .nt knowledge graph and its surface forms.
2. Search Wikidata for the closest item, e.g. "Heart" -> Q1072. Two tiers:
   an exact SPARQL label/alias match first (cheap - a single indexed literal
   lookup, no fuzzy scoring), then the wbsearchentities API (fuzzy, best of
   up to 5 candidates per language) if that misses. If both miss and
   --abbreviations was given, retry both tiers against the text with any
   known abbreviation expanded (e.g. "...sustained VA" -> "...sustained
   ventricular arrhythmia") - see src/prepare_wikidata_search.py.
3. Fetch its properties, then follow related treatments and retrieve drug details.
4. Add the returned facts and connect them to the surface-form node.
5. Save a NEW .nt graph and a JSON report.

Install: pip install rdflib requests
Run:     py -3.12 enrich_wikidata.py --source api --max-related-treatments 30 \\
             --abbreviations wikidata_search_input.json

--source api:    use the API for both search and property retrieval; also
                 skips the exact-match SPARQL search tier entirely.
--source sparql: use the API for search, then SPARQL for properties.
--source auto:   try SPARQL; after a failure, use the API for the rest of the run.

Related-treatment expansion uses the API in every mode (one extra hop).

Search uses English/German labels and aliases. The similarity score compares
text; it does not verify the medical meaning. Missing values stay absent.
"""

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
import time
from pathlib import Path
from urllib.parse import quote, unquote

import requests
from rdflib import Graph, Literal, Namespace, URIRef, RDFS, BNode, RDF
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03-json-knowledge-graph" / "src"))
from connect_graph_navigation import connect_graph_navigation  # noqa: E402

DS = Namespace("http://digistrucmed.org/")
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"


LANGUAGES = ("en", "de")
PROPERTIES = {
    "instance_of": "P31",
    "subclass_of": "P279",
    "has_effect": "P1542",
    "studied_by": "P2579",
    "drug_or_therapy_used_for_treatment": "P2176",
    "medical_condition_treated": "P2175",
    "drugbank_id": "P715",
    
}
DRUG_PROPERTIES = {
    "atc_code": "P267", "route_of_administration": "P636",
    "significant_drug_interaction": "P769", "chemical_formula": "P274",
    "cas_registry_number": "P231", "pubchem_cid": "P662",
    "subject_has_role": "P2868", "side_effect": "P1909", "medical_conditional_treated": "P2175", "drug_interaction": "P769", "pharmacological_class": "P1995", "mechanism_of_action": "P769"
}

ALL_PROPERTIES = {**PROPERTIES, **DRUG_PROPERTIES}
WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")
API_ENDPOINT = "https://www.wikidata.org/w/api.php"


# Use the same enum-to-clinical-class table as the RML mapping. Lives in
# 03-json-knowledge-graph/ (that pipeline's own mapping consumes it too),
# not here - this script only borrows it across the step boundary.
CLASS_LOOKUP = Path(__file__).resolve().parent.parent / "03-json-knowledge-graph" / "mappings" / "ontology-class-lookup.json"
CONCEPT_CLASSES = {
    row["code"]: URIRef(row["class_iri"])
    for row in json.loads(CLASS_LOOKUP.read_text(encoding="utf-8"))["entity_concept_types"]
}


# STEP 1: READ SURFACE FORMS from the input knowledge graph.
def collect_surface_forms(graph):
    """Read /entity/Heart as Heart, even when no rdfs:label exists."""
    forms = {}
    prefix = str(DS) + "entity/"
    for value in graph.objects(None, DS.surfaceForm):
        if isinstance(value, URIRef) and str(value).startswith(prefix):
            text = unquote(str(value)[len(prefix):])
            term = value  # Keep the existing IRI exactly as written.
        elif isinstance(value, Literal):
            text = str(value)
            term = URIRef(prefix + quote(text, safe=""))
        elif isinstance(value, URIRef):
            label = graph.value(value, RDFS.label)
            if not isinstance(label, Literal):
                continue
            text, term = str(label), value
        else:
            continue
        if text.strip():
            forms.setdefault(text, set()).add(term)
    return forms


# STEP 1C (optional): ABBREVIATION EXPANSIONS -- src/prepare_wikidata_search.py's output.
def load_expansions(path):
    """{surface_form: expanded_surface_form} for rows that had one - a search
    fallback when the original text (which may contain an abbreviation like
    "VA") finds nothing on its own."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["surface_form"]: row["expanded_surface_form"]
            for row in rows if row.get("expanded_surface_form")}


SCIENTIFIC_ARTICLE_TERMS = (
    "scientific article", "scholarly article", "research article", "wissenschaftlicher artikel")


# STEP 1B: EXACT MATCH -- a cheap SPARQL literal lookup, tried before the API.
def _sparql_escape(text):
    """Escape for a SPARQL short string literal ("..."). Backslash and quote
    were the only two escaped here originally; found the hard way that's not
    enough - some extracted surface forms/candidates contain a literal
    newline (e.g. "15\\nmg/kg/day", a PDF line-wrap artifact), and a short
    string literal cannot contain a raw newline/CR/tab per the SPARQL/Turtle
    grammar - embedding one breaks query parsing entirely (live HTTP 400 Bad
    Request from the Wikidata query service, not a graceful per-row miss)."""
    return (text.replace("\\", "\\\\").replace('"', '\\"')
            .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))


def build_exact_match_query(text):
    """An item's rdfs:label/skos:altLabel bound to a literal is an indexed
    lookup on Wikidata's backend (Blazegraph) - no scan, no fuzzy scoring
    needed, since either some item's label is exactly this text or it isn't.
    Far cheaper than the wbsearchentities API call below when it hits."""
    escaped = _sparql_escape(text)
    clauses = " UNION ".join(
        f'{{ ?item rdfs:label "{escaped}"@{language} }} UNION {{ ?item skos:altLabel "{escaped}"@{language} }}'
        for language in LANGUAGES)
    return f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX schema: <http://schema.org/>
SELECT DISTINCT ?item ?itemLabel ?itemDescription WHERE {{
  {{ {clauses} }}
  OPTIONAL {{ ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel) = "en") }}
  OPTIONAL {{ ?item schema:description ?itemDescription . FILTER(LANG(?itemDescription) = "en") }}
}}
LIMIT 10
"""


def search_wikidata_exact(session, text):
    """Return the first non-article item with an exact-text label/alias, or
    None. Ambiguous exact matches (rare for a full phrase, more likely for a
    short abbreviation) are not disambiguated further - just the first
    survivor after the scientific-article filter, same policy as the API
    path uses for its own tie-breaks."""
    response = session.get(SPARQL_ENDPOINT, params={
        "query": build_exact_match_query(text), "format": "json",
    }, timeout=(10, 25))
    response.raise_for_status()
    for row in response.json()["results"]["bindings"]:
        description = row.get("itemDescription", {}).get("value", "").casefold()
        if any(term in description for term in SCIENTIFIC_ARTICLE_TERMS):
            continue
        item_id = row["item"]["value"].removeprefix("http://www.wikidata.org/entity/")
        if not re.fullmatch(r"Q[1-9][0-9]*", item_id):
            continue
        return {"id": item_id, "label": row.get("itemLabel", {}).get("value", text),
                "description": row.get("itemDescription", {}).get("value", ""),
                "language": "en", "similarity": 1.0, "search_rank": 1, "match_tier": "exact"}
    return None


# STEP 2: SEARCH API -- find a QID from text.
def search_wikidata_api(session, surface_form):
    """API request #1: wbsearchentities searches names, not RDF triples."""
    candidates = {}
    normalized = " ".join(surface_form.casefold().split())
    for language in LANGUAGES:
        # This line sends the SEARCH request to www.wikidata.org/w/api.php.
        response = session.get(API_ENDPOINT, params={
            "action": "wbsearchentities", "format": "json", "type": "item",
            "search": surface_form, "language": language,
            "uselang": language, "limit": 5,
        }, headers={"Accept": "application/json"}, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise ValueError(str(payload["error"]))
        for rank, hit in enumerate(payload["search"], 1):
            # A paper about a procedure is not the procedure itself.
            description = hit.get("description", "").casefold()
            if any(term in description for term in SCIENTIFIC_ARTICLE_TERMS):
                continue
            names = [hit.get("label", ""), hit.get("match", {}).get("text", "")]
            score = max(SequenceMatcher(None, normalized,
                        " ".join(name.casefold().split())).ratio() for name in names)
            candidate = {"id": hit["id"], "label": hit.get("label", ""),
                         "description": hit.get("description", ""),
                         "language": language, "similarity": round(score, 4),
                         "search_rank": rank, "match_tier": "api"}
            old = candidates.get(hit["id"])
            if old is None or (score, -rank) > (old["similarity"], -old["search_rank"]):
                candidates[hit["id"]] = candidate
    ranked = sorted(candidates.values(), key=lambda c: (-c["similarity"], c["search_rank"], c["id"]))
    return ranked[0] if ranked else None


def find_wikidata_match(session, text, try_exact):
    """Two tiers for one piece of text: exact SPARQL label match, then fuzzy
    API search. try_exact is False once SPARQL has failed once this run (see
    active_source in main()) or --source api was requested."""
    if try_exact:
        selection = search_wikidata_exact(session, text)
        if selection:
            return selection
    return search_wikidata_api(session, text)


# STEP 3A: SPARQL -- construct a query, then send it to the query endpoint.
def build_sparql_query(item_id):
    """Fetch all requested direct values and metadata for the item and targets."""
    if not re.fullmatch(r"Q[1-9][0-9]*", item_id):
        raise ValueError(f"Invalid Wikidata item ID: {item_id}")
    properties = " ".join("wdt:" + pid for pid in ALL_PROPERTIES.values())
    languages = ", ".join(json.dumps(lang) for lang in LANGUAGES)
    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX schema: <http://schema.org/>
SELECT DISTINCT ?s ?p ?o WHERE {{
  {{
    BIND(wd:{item_id} AS ?s)
    VALUES ?p {{ {properties} }}
    ?s ?p ?o .
  }} UNION {{
    {{ BIND(wd:{item_id} AS ?s) }} UNION {{
      VALUES ?relation {{ {properties} }}
      wd:{item_id} ?relation ?s .
      FILTER(ISIRI(?s))
    }}
    VALUES ?p {{ rdfs:label skos:altLabel schema:description  }}
    ?s ?p ?o .
    FILTER(LANG(?o) IN ({languages}))
  }}
}}
"""


def fetch_with_sparql(session, item_id):
    """This is where the SPARQL query is actually sent over the network."""
    response = session.get(SPARQL_ENDPOINT, params={
        "query": build_sparql_query(item_id), "format": "json",
    }, timeout=(10, 25))
    response.raise_for_status()
    # Convert SPARQL JSON into ordinary RDF (subject, predicate, object) tuples.
    triples = []
    blank_nodes = {}
    for row in response.json()["results"]["bindings"]:
        subject = sparql_value_to_rdf(row["s"], blank_nodes)
        predicate = sparql_value_to_rdf(row["p"], blank_nodes)
        obj = sparql_value_to_rdf(row["o"], blank_nodes)
        if not isinstance(subject, (URIRef, BNode)) or not isinstance(predicate, URIRef):
            raise ValueError("Invalid RDF subject or predicate in SPARQL response")
        triples.append((subject, predicate, obj))
    return triples


def sparql_value_to_rdf(value, blank_nodes):
    """An IRI becomes URIRef; text becomes Literal (keeping its language)."""
    if value["type"] == "uri":
        return URIRef(value["value"])
    if value["type"] == "bnode":
        return blank_nodes.setdefault(value["value"], BNode())
    if value["type"] in {"literal", "typed-literal"}:
        return Literal(value["value"], lang=value.get("xml:lang"),
                       datatype=value.get("datatype"))
    raise ValueError(f"Unsupported SPARQL value type: {value['type']}")


# STEP 3B: ENTITY API -- an alternative way to retrieve the same facts.
def request_entities_api(session, ids, claims=False):
    """API request #2: wbgetentities downloads item data as JSON."""
    entities = {}
    ids = sorted(set(ids))
    for start in range(0, len(ids), 50):
        response = session.get(API_ENDPOINT, params={
            "action": "wbgetentities", "format": "json",
            "ids": "|".join(ids[start:start + 50]),
            "props": "labels|descriptions|aliases" + ("|claims" if claims else ""),
            "languages": "|".join(LANGUAGES),
        }, headers={"Accept": "application/json"}, timeout=(10, 45))
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise ValueError(str(payload["error"]))
        entities.update(payload["entities"])
    return entities


def fetch_with_api(session, item_id):
    """Return RDF triples from API JSON, without using the SPARQL endpoint.

    Use preferred claims when present, otherwise normal claims, matching
    Wikidata direct-property (wdt:) rank selection. No-value claims add no edge.
    """
    entities = request_entities_api(session, [item_id], claims=True)
    entity = entities[item_id]
    if "missing" in entity:
        raise ValueError(f"Wikidata item {item_id} is missing")
    triples = []
    related = set()
    base = "http://www.wikidata.org/entity/"

    for pid in ALL_PROPERTIES.values():
        claims = entity.get("claims", {}).get(pid, [])
        rank = "preferred" if any(c.get("rank") == "preferred" for c in claims) else "normal"
        for claim in claims:
            if claim.get("rank") != rank:
                continue
            snak = claim["mainsnak"]
            if snak["snaktype"] == "novalue":
                continue
            if snak["snaktype"] == "somevalue":
                obj = BNode()  # Wikidata says a value exists but is unknown.
            else:
                data = snak["datavalue"]
                if data["type"] == "wikibase-entityid":
                    target = data["value"]["id"]
                    related.add(target)
                    obj = URIRef(base + target)
                elif data["type"] == "string":
                    obj = Literal(data["value"])  # For example, a DrugBank ID.
                else:
                    raise ValueError(f"Unexpected value type for {pid}: {data['type']}")
            predicate = URIRef("http://www.wikidata.org/prop/direct/" + pid)
            triples.append((URIRef(base + item_id), predicate, obj))

    entities.update(request_entities_api(session, related - {item_id}))
    metadata = {"labels": str(RDFS.label), "descriptions": "http://schema.org/description",
                "aliases": "http://www.w3.org/2004/02/skos/core#altLabel"}
    for qid, data in entities.items():
        for field, predicate in metadata.items():
            for language in LANGUAGES:
                values = data.get(field, {}).get(language, [])
                if isinstance(values, dict):
                    values = [values]
                for value in values:
                    text = Literal(value["value"], lang=value["language"])
                    triples.append((URIRef(base + qid), URIRef(predicate), text))
    return triples


def fetch_properties(session, item_id, source):
    """Return (triples, source_used, fallback_reason). No hidden state."""
    if source == "api":
        return fetch_with_api(session, item_id), "api", None
    try:
        return fetch_with_sparql(session, item_id), "sparql", None
    except (requests.RequestException, ValueError, KeyError) as error:
        if source == "sparql":
            raise
        response = getattr(error, "response", None)
        reason = f"HTTP {response.status_code}" if response is not None else type(error).__name__
        print(f"  SPARQL failed ({reason}); switching to entity API.", flush=True)
        # main() remembers this switch even if the following API request fails.
        return None, "api", reason


# STEP 3C: Find treatments, then visit each treatment to retrieve drug details.
def search_treatments_api(session, condition_id, limit, class_members=False):
    """API reverse search: drugs/therapies with P2175 pointing to this condition."""
    query = f"haswbstatement:P2175={condition_id}"
    if class_members:
        # A drug class (e.g. beta blocker) has no DrugBank ID of its own.
        query = (f"haswbstatement:P31={condition_id}|P279={condition_id}|P2868={condition_id} "
                 "haswbstatement:P715")
    ids = []
    continuation = {}
    while len(ids) < limit:
        response = session.get(API_ENDPOINT, params={
            "action": "query", "format": "json", "list": "search",
            "srsearch": query,
            "srnamespace": 0, "srlimit": min(50, limit - len(ids)),
            **continuation,
        }, headers={"Accept": "application/json"}, timeout=(10, 45))
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise ValueError(str(payload["error"]))
        ids.extend(hit["title"] for hit in payload["query"]["search"]
                   if re.fullmatch(r"Q[1-9][0-9]*", hit["title"]))
        continuation = payload.get("continue", {})
        if not continuation:
            break
    return ids, bool(continuation)


def fetch_related_treatments(session, item_id, item_triples, cache, limit=30):
    """One extra hop: condition -> drug/therapy -> DrugBank ID and other facts.

    P2176 also includes procedures, so we do not label every result a drug.
    Drug interactions and other conditions are not recursively expanded.
    """
    item = WD[item_id]
    forward_ids = sorted({str(obj).removeprefix(str(WD))
                          for subject, predicate, obj in item_triples
                          if subject == item and predicate == WDT.P2176
                          and isinstance(obj, URIRef) and str(obj).startswith(str(WD))})
    report = {"expanded_items": {}, "failed_items": {}, "reverse_search_error": None,
              "limit": limit, "truncated": len(forward_ids) > limit}
    try:
        reverse_ids, more = search_treatments_api(session, item_id, limit)
        report["truncated"] |= more
    except (requests.RequestException, ValueError, KeyError) as error:
        reverse_ids = []
        report["reverse_search_error"] = str(error) or type(error).__name__
    # Also find individual drugs that are members/subclasses of this drug
    # class, or have it as a pharmacological role. Search requires a P715 ID.
    report["class_members_search_error"] = None
    try:
        member_ids, more = search_treatments_api(session, item_id, limit, class_members=True)
        report["truncated"] |= more
    except (requests.RequestException, ValueError, KeyError) as error:
        member_ids = []
        report["class_members_search_error"] = str(error) or type(error).__name__
    candidates = [qid for qid in dict.fromkeys(forward_ids + reverse_ids + member_ids) if qid != item_id]
    report["truncated"] |= len(candidates) > limit
    additions = []
    for qid in candidates[:limit]:
        try:
            # Visit the drug/therapy item itself, not just its label.
            if qid not in cache:
                cache[qid] = fetch_with_api(session, qid)
            triples = cache[qid]
            treatment = WD[qid]
            # Confirm reverse-search hits against current selected-rank claims.
            class_relation = next((predicate for predicate in (WDT.P31, WDT.P279, WDT.P2868)
                                   if (treatment, predicate, item) in triples), None)
            has_drugbank = any(s == treatment and p == WDT.P715 and isinstance(o, Literal)
                               for s, p, o in triples)
            if qid in forward_ids:
                found_by = "P2176"
            elif (treatment, WDT.P2175, item) in triples:
                found_by = "inverse P2175"
            elif class_relation is not None and has_drugbank:
                found_by = "drug class member: " + str(class_relation).removeprefix(str(WDT))
            else:
                continue
            additions.extend(triples)
            report["expanded_items"][qid] = {
                "labels": {obj.language: str(obj) for subject, predicate, obj in triples
                           if subject == treatment and predicate == RDFS.label and isinstance(obj, Literal)},
                "drugbank_ids": sorted({str(obj) for subject, predicate, obj in triples
                                        if subject == treatment and predicate == WDT.P715 and isinstance(obj, Literal)}),
                "found_by": found_by,
                "data_source": "api",
            }
        except (requests.RequestException, ValueError, KeyError) as error:
            report["failed_items"][qid] = str(error) or type(error).__name__
    return additions, report


def local_iri(source):
    """Keep Wikidata URI paths distinct; encode other complete URIs losslessly."""
    prefix = "http://www.wikidata.org/"
    if source.startswith(prefix):
        return URIRef(str(DS) + "wikidata/" + source[len(prefix):])
    return URIRef(str(DS) + "wikidata/external/" + quote(source, safe=""))


# STEP 4: CONNECT WIKIDATA FACTS TO YOUR KNOWLEDGE GRAPH.
def add_to_knowledge_graph(graph, surface_nodes, match, triples):
    """Example of the connection added here:

    ds:entity/Heart --ds:wikidataMatch--> ds:wikidata/entity/Q1072
    ds:wikidata/entity/Q1072 --ds:instance_of--> another local Wikidata node
    ds:wikidata/entity/Q1072 --ds:hasdescription--> "..."@en

    The real .nt file uses full IRIs; these abbreviations are for readability.
    """
    additions = Graph()
    original_item = URIRef("http://www.wikidata.org/entity/" + match["id"])
    local_item = copy_resource(original_item, additions)
    additions.add((local_item, DS.wikidataSource, original_item))
    for node in surface_nodes:
        additions.add((node, DS.wikidataMatch, local_item))
        additions.add((node, DS.wikidataMatchSimilarity, Literal(match["similarity"])))
        # Reuse the clinical type provided by JSON/RML, including older graphs
        # that recorded the enum only on the mention. Do not make a Wikidata
        # concept a document-local ExtractedEntity/MedicationEntity.
        types = set(graph.objects(node, RDF.type)) & set(CONCEPT_CLASSES.values())
        for mention in graph.subjects(DS.surfaceForm, node):
            for code in graph.objects(mention, DS.entityType):
                cls = CONCEPT_CLASSES.get(str(code).removeprefix(str(DS) + "entity-type/"))
                if cls:
                    types.add(cls)
        for cls in types:
            additions.add((node, RDF.type, cls))
            additions.add((local_item, RDF.type, cls))
            additions.add((local_item, DS.classificationSource, node))

    # Translate Wikidata predicates into readable digistrucmed predicates.
    predicate_names = {}
    for name, pid in ALL_PROPERTIES.items():
        predicate_names["http://www.wikidata.org/prop/direct/" + pid] = DS[name]
    predicate_names[str(RDFS.label)] = DS.label
    predicate_names["http://www.w3.org/2004/02/skos/core#altLabel"] = DS.alias
    predicate_names["http://schema.org/description"] = DS.hasdescription

    for subject, predicate, obj in triples:
        local_subject = copy_resource(subject, additions)
        local_predicate = predicate_names.get(str(predicate), local_iri(str(predicate)))
        additions.add((local_predicate, DS.wikidataSource, predicate))
        local_object = copy_resource(obj, additions)
        additions.add((local_subject, local_predicate, local_object))
        if predicate == RDFS.label:
            # Standard label attributes let graph viewers show "bisoprolol".
            additions.add((local_subject, RDFS.label, obj))
        if predicate == WDT.P715 and isinstance(obj, Literal) and str(obj).strip():
            additions.add((local_subject, RDF.type, DS.Drug))
            additions.add((local_subject, DS.drugBankId, obj))
        if predicate == WDT.P2175 and isinstance(obj, URIRef):
            # The object is explicitly a treated medical condition.
            additions.add((local_object, RDF.type, DS.PatientCondition))

        # Also put the selected item's description directly on /entity/Heart.
        if subject == original_item and local_predicate == DS.hasdescription:
            for node in surface_nodes:
                additions.add((node, DS.hasdescription, obj))

    # Prefer an English name, then German; preserve every bilingual label too.
    if not list(additions.objects(local_item, RDFS.label)) and match.get("label"):
        additions.add((local_item, RDFS.label, Literal(match["label"], lang=match.get("language", "en"))))
    for resource in set(additions.subjects(RDFS.label, None)):
        labels = list(additions.objects(resource, RDFS.label))
        preferred = min(labels, key=lambda value: (
            LANGUAGES.index(getattr(value, "language", ""))
            if getattr(value, "language", "") in LANGUAGES
            else len(LANGUAGES),
            str(value)))
        additions.add((resource, DS.name, Literal(str(preferred))))
        if (resource, RDF.type, DS.Drug) in additions:
            additions.add((resource, DS.drugName, Literal(str(preferred))))
    graph += additions  # This is where the new triples enter the input graph.


def copy_resource(value, additions):
    """Localize IRIs and keep source links; leave text/language tags unchanged."""
    if not isinstance(value, URIRef):
        return value
    local = local_iri(str(value))
    additions.add((local, DS.wikidataSource, value))
    qid = str(value).removeprefix(str(WD))
    if re.fullmatch(r"[QP][1-9][0-9]*", qid):
        additions.add((local, DS.wikidataId, Literal(qid)))
        if qid.startswith("Q"):
            additions.add((local, RDF.type, DS.WikidataEntity))
    return local


def create_session():
    """HTTP setup: retry API failures, but let SPARQL fall back immediately.

    The SPARQL endpoint is deliberately mounted with max_retries=0, separate
    from the generic-https:// retry policy above: this module's own
    search_wikidata_exact() is one tier of a per-term exact-then-fuzzy-API
    search, so a SPARQL failure should fail fast and let the caller move to
    the next tier immediately, not spend time retrying. That's WRONG for a
    caller with no such fallback tier - src/search_wikidata_candidates.py
    and src/expand_wikidata_entities.py each send a single batched SPARQL
    request that IS the whole operation, no fallback to jump to - see
    sparql_request() below, which those two scripts use instead of a bare
    session.get()/post()."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "GuideLineKG-WikidataEnrichment/1.0 (research script)",
        "Accept": "application/sparql-results+json",
    })
    retry = Retry(total=3, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], respect_retry_after_header=True)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount(SPARQL_ENDPOINT, HTTPAdapter(max_retries=0))
    return session


SPARQL_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
SPARQL_MAX_RETRIES = 5
SPARQL_BATCH_PACING_SECONDS = 1.0


def sparql_request(session, method, **kwargs):
    """POST/GET to SPARQL_ENDPOINT with its own retry+backoff, for callers
    that have no fallback tier to jump to on failure (see create_session's
    docstring for why the session itself won't retry SPARQL calls). Handles
    three distinct failure shapes the same way: an HTTP response with a
    retryable status code (429/5xx); a connection-level failure with no
    response at all (requests.exceptions.RequestException - reproduced live
    as "RemoteDisconnected: Remote end closed connection without response",
    immediately after a 429, i.e. the query service dropped the connection
    outright rather than responding to a client it had just rate-limited);
    and a 200 response whose body is truncated/malformed JSON (reproduced
    live under a loaded WDQS as a JSONDecodeError mid-document - the request
    itself succeeded, only the body got corrupted in transit, so it's
    retried exactly like the other two, not treated as a permanent parse
    error). Checking response.status_code alone misses the second and third
    cases entirely - the second has no response object to check, and the
    third has a perfectly fine (200) status."""
    response = None
    json_error = None
    for attempt in range(SPARQL_MAX_RETRIES):
        try:
            response = getattr(session, method)(SPARQL_ENDPOINT, timeout=(10, 60), **kwargs)
        except requests.exceptions.RequestException as error:
            wait = 2.0 ** attempt
            print(f"  {type(error).__name__}, retrying in {wait:.0f}s "
                  f"({attempt + 1}/{SPARQL_MAX_RETRIES})...", flush=True)
            time.sleep(wait)
            continue
        if response.status_code in SPARQL_RETRYABLE_STATUS:
            wait = float(response.headers.get("Retry-After", 2 ** attempt))
            print(f"  {response.status_code}, retrying in {wait:.0f}s "
                  f"({attempt + 1}/{SPARQL_MAX_RETRIES})...", flush=True)
            time.sleep(wait)
            continue
        try:
            response.json()
        except requests.exceptions.JSONDecodeError as error:
            json_error = error
            wait = 2.0 ** attempt
            print(f"  malformed JSON response, retrying in {wait:.0f}s "
                  f"({attempt + 1}/{SPARQL_MAX_RETRIES})...", flush=True)
            time.sleep(wait)
            continue
        break
    else:
        if json_error is not None:
            raise RuntimeError(f"Request returned malformed JSON after {SPARQL_MAX_RETRIES} retries") from json_error
    if response is None:
        raise RuntimeError(f"Request failed after {SPARQL_MAX_RETRIES} connection-error retries")
    response.raise_for_status()
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent.parent / "03-json-knowledge-graph" / "output_json" / "knowledge_graph_json.nt",
                        help="Existing N-Triples (.nt) graph")
    parser.add_argument("-o", "--output", type=Path, help="Default: INPUT_wikidata.nt")
    parser.add_argument("--source", choices=["auto", "api", "sparql"], default="auto",
                        help="auto: fall back to entity API; api: bypass SPARQL")
    parser.add_argument("--max-related-treatments", type=int, default=30,
                        help="Maximum related drug/therapy items visited per match (default: 30)")
    parser.add_argument("--abbreviations", type=Path, default=None,
                         help="src/prepare_wikidata_search.py output; retry a failed search with "
                              "the abbreviation-expanded text (e.g. 'VA' -> 'ventricular arrhythmia')")
    args = parser.parse_args()
    if args.max_related_treatments < 1:
        parser.error("--max-related-treatments must be at least 1")
    output = args.output or args.input.with_name(args.input.stem + "_wikidata.nt")
    if output.resolve() == args.input.resolve():
        parser.error("Choose an output different from the input")

    # 1. Load your existing graph and read ds:surfaceForm values.
    graph = Graph().parse(args.input, format="nt")
    original_count = len(graph)
    # One query per distinct text, even when many entities share that text.
    surface_forms = collect_surface_forms(graph)
    expansions = load_expansions(args.abbreviations) if args.abbreviations else {}

    failures = {}
    matched = 0
    selections = {}
    item_cache = {}
    treatment_cache = {}
    expansion_cache = {}
    treatment_reports = {}
    active_source = args.source
    fallback_reason = None
    with create_session() as session:
        for index, (text, terms) in enumerate(sorted(surface_forms.items()), 1):
            print(f"[{index}/{len(surface_forms)}] {text!r}", flush=True)
            try:
                # 2. SEARCH: exact SPARQL label match first (cheap), then the
                # fuzzy API search; if both miss and the text contains a
                # known abbreviation, retry both tiers against the expanded
                # text (e.g. "pregnant patient with sustained VA" ->
                # "...ventricular arrhythmia").
                try_exact = args.source != "api" and active_source != "api"
                try:
                    selection = find_wikidata_match(session, text, try_exact)
                except (requests.RequestException, ValueError, KeyError) as error:
                    print(f"  Exact-match SPARQL failed ({error}); using API search only.", flush=True)
                    active_source = "api"
                    selection = search_wikidata_api(session, text)
                if not selection and text in expansions:
                    expanded_text = expansions[text]
                    print(f"  No match for {text!r}; retrying with {expanded_text!r}", flush=True)
                    try_exact = args.source != "api" and active_source != "api"
                    try:
                        selection = find_wikidata_match(session, expanded_text, try_exact)
                    except (requests.RequestException, ValueError, KeyError) as error:
                        print(f"  Exact-match SPARQL failed ({error}); using API search only.", flush=True)
                        active_source = "api"
                        selection = search_wikidata_api(session, expanded_text)
                    if selection:
                        selection["expanded_from"] = text
                if selection:
                    item_id = selection["id"]
                    if item_id not in item_cache:
                        # 3. DATA REQUEST: QID -> RDF triples (SPARQL or API).
                        triples, source, reason = fetch_properties(session, item_id, active_source)
                        if reason:
                            active_source = "api"  # Avoid more SPARQL timeouts this run.
                            fallback_reason = reason
                            triples = fetch_with_api(session, item_id)
                        item_cache[item_id] = (triples, source)
                    triples, source = item_cache[item_id]
                    selection["data_source"] = source
                    # Follow treatment links and retrieve each drug's identifiers.
                    if item_id not in expansion_cache:
                        expansion_cache[item_id] = fetch_related_treatments(
                            session, item_id, triples, treatment_cache, args.max_related_treatments)
                    treatment_triples, treatment_report = expansion_cache[item_id]
                    treatment_reports[text] = treatment_report
                    print(f"  Expanded {len(treatment_report['expanded_items'])} related drug/therapy item(s)", flush=True)
                    if treatment_report["failed_items"] or treatment_report["reverse_search_error"] or treatment_report["class_members_search_error"]:
                        print("  Treatment expansion is partial; see the report.", flush=True)
                    if treatment_report["truncated"]:
                        print("  Treatment limit reached; increase --max-related-treatments for more.", flush=True)
                    # 4. Add facts and link this surface-form node to the item.
                    add_to_knowledge_graph(graph, terms, selection, triples + treatment_triples)
                    selections[text] = selection
                    matched += 1
                    print(f"  {item_id}: {selection['label']} (similarity {selection['similarity']})", flush=True)
                else:
                    print("  No search match", flush=True)
            except (requests.RequestException, ValueError, KeyError) as error:
                failures[text] = str(error)
                print(f"  Query failed: {error}", flush=True)
            time.sleep(1)  # Send requests sequentially and give the endpoint a break.

    # 5. Save a new graph and a report; the input file stays unchanged.
    output.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(output), format="nt", encoding="utf-8")
    drugbank_identifiers = sorted(
        ({"drug_iri": str(drug), "drugbank_id": str(identifier)}
         for drug, identifier in graph.subject_objects(DS.drugbank_id)),
        key=lambda row: (row["drug_iri"], row["drugbank_id"]))
    print(f"DrugBank identifiers in saved graph: {len(drugbank_identifiers)}", flush=True)
    report = output.with_suffix(".report.json")
    report.write_text(json.dumps({
        "surface_forms": len(surface_forms), "matched": matched,
        "sparql_fallback_reason": fallback_reason,
        "related_treatments": treatment_reports,
        "drugbank_identifiers": drugbank_identifiers,
        "unmatched": len(surface_forms) - matched - len(failures),
        "selected_matches": selections, "failed_queries": failures, "added_triples": len(graph) - original_count,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {output} ({len(graph) - original_count} added triples)")
    print(f"Report: {report}")
    if failures or any(r["failed_items"] or r["reverse_search_error"] or r["class_members_search_error"] for r in treatment_reports.values()):
        print("Some queries failed; the saved graph is partially enriched.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Enrich the KG's UMLS concepts with Metathesaurus relations (MRREL, via the
UTS REST API's /relations endpoint) and authoritative semantic types (MRSTY,
via the /CUI/{cui} endpoint).

Reads the unique CUI set from data/umls_candidates.csv (every CUI already
materialized as a ds:UMLSConcept in the graph) and fetches each CUI exactly
once, regardless of how many mentions/candidates reference it. Results are
cached to disk per CUI (data/umls_cache/{cui}.json) so re-running this script
only fetches CUIs that are new since the last run.

Writes (URIs precomputed here, not built in RML, per the pipeline's
"no string transformation in RML" rule):
  data/umls_relations.csv      - cui1, cui1_uri, rel, rela, cui2, cui2_uri,
                                  cui2_name, sab, relation_uri
  data/umls_semantic_types.csv - cui, cui_uri, tui, semantic_type_name

cui2/cui2_uri are taken verbatim from the API's relatedId and are not
guaranteed to be global UMLS CUIs: MRREL "relations" can be source-asserted
against a source-vocabulary-local code (e.g. an RXCUI) rather than a merged
CUI. cui2_uri is still minted via the same ds:umls/{id} template either way -
that is a valid RDF node reference even when nothing else in this graph
describes it further.

Requires UMLS_API_KEY (free from https://uts.nlm.nih.gov), read from the
environment or a .env file next to this pipeline (kg_build/.env). The key is
never logged or written to any output file.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from dotenv import load_dotenv

NS = "http://digistrucmed.org/"
API_BASE = "https://uts-ws.nlm.nih.gov/rest"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("fetch_umls_relations")


def uri(local_type: str, local_id: str) -> str:
    return f"{NS}{local_type}/{quote(local_id, safe='')}"


def unique_cuis(data_dir: Path) -> list[str]:
    cuis: set[str] = set()
    with open(data_dir / "umls_candidates.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cui = row.get("cui", "").strip()
            if cui:
                cuis.add(cui)
    return sorted(cuis)


def cui_from_related_id(related_id: str) -> str:
    # relatedId looks like "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0004238"
    return related_id.rstrip("/").rsplit("/", 1)[-1]


def fetch_json(session: requests.Session, url: str, params: dict[str, Any],
                max_retries: int = 5) -> dict[str, Any] | None:
    delay = 1.0
    for attempt in range(max_retries):
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None
        if resp.status_code in (429, 500, 502, 503):
            log.warning("HTTP %d on %s (attempt %d/%d), backing off %.1fs",
                        resp.status_code, url, attempt + 1, max_retries, delay)
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
    log.error("Giving up on %s after %d retries", url, max_retries)
    return None


def fetch_cui_relations(session: requests.Session, api_key: str, cui: str,
                         version: str) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    page = 1
    while True:
        data = fetch_json(
            session, f"{API_BASE}/content/{version}/CUI/{cui}/relations",
            {"apiKey": api_key, "pageNumber": page},
        )
        if not data or not data.get("result"):
            break
        result = data["result"]
        if isinstance(result, str) or not result:
            break
        relations.extend(result)
        page_count = data.get("pageCount", 1)
        if page >= page_count:
            break
        page += 1
    return relations


def fetch_cui_semantic_types(session: requests.Session, api_key: str, cui: str,
                              version: str) -> list[dict[str, Any]]:
    data = fetch_json(session, f"{API_BASE}/content/{version}/CUI/{cui}",
                       {"apiKey": api_key})
    if not data or not data.get("result"):
        return []
    return data["result"].get("semanticTypes", [])


def load_or_fetch(session: requests.Session, api_key: str, cui: str, version: str,
                   cache_dir: Path, rate_limit_delay: float, refresh: bool) -> dict[str, Any]:
    cache_path = cache_dir / f"{cui}.json"
    if cache_path.exists() and not refresh:
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    relations = fetch_cui_relations(session, api_key, cui, version)
    time.sleep(rate_limit_delay)
    semantic_types = fetch_cui_semantic_types(session, api_key, cui, version)
    time.sleep(rate_limit_delay)

    record = {"cui": cui, "relations": relations, "semantic_types": semantic_types}
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return record


def build(data_dir: Path, cache_dir: Path, version: str, rate_limit_delay: float,
          refresh: bool) -> None:
    load_dotenv(Path(__file__).parent.parent / ".env")
    api_key = os.environ.get("UMLS_API_KEY")
    if not api_key:
        raise SystemExit("UMLS_API_KEY not set (checked environment and kg_build/.env)")

    cuis = unique_cuis(data_dir)
    log.info("%d unique CUIs to enrich", len(cuis))

    session = requests.Session()
    relation_rows: list[dict[str, Any]] = []
    semtype_rows: list[dict[str, Any]] = []
    seen_semtypes: set[tuple[str, str]] = set()
    failed: list[str] = []

    for i, cui in enumerate(cuis, 1):
        try:
            record = load_or_fetch(session, api_key, cui, version, cache_dir,
                                    rate_limit_delay, refresh)
        except requests.RequestException as exc:
            log.error("Failed to fetch %s: %s", cui, exc)
            failed.append(cui)
            continue

        for rel in record["relations"]:
            related_id = rel.get("relatedId", "")
            if not related_id:
                continue
            rel_label = rel.get("relationLabel", "")
            rela = rel.get("additionalRelationLabel", "")
            sab = rel.get("rootSource", "")
            cui2 = cui_from_related_id(related_id)
            relation_local = "_".join([cui, rel_label or "NA", rela or "NA", cui2, sab or "NA"])
            relation_rows.append({
                "cui1": cui,
                "cui1_uri": uri("umls", cui),
                "rel": rel_label,
                "rela": rela,
                "cui2": cui2,
                "cui2_uri": uri("umls", cui2),
                "cui2_name": rel.get("relatedIdName", ""),
                "sab": sab,
                "relation_uri": uri("umls-relation", relation_local),
            })

        for st in record["semantic_types"]:
            tui = st.get("uri", "").rstrip("/").rsplit("/", 1)[-1]
            name = st.get("name", "")
            key = (cui, tui)
            if tui and key not in seen_semtypes:
                seen_semtypes.add(key)
                semtype_rows.append({
                    "cui": cui,
                    "cui_uri": uri("umls", cui),
                    "tui": tui,
                    "semantic_type_name": name,
                })

        if i % 10 == 0 or i == len(cuis):
            log.info("Processed %d/%d CUIs", i, len(cuis))

    _write_csv(data_dir / "umls_relations.csv", relation_rows,
               ["cui1", "cui1_uri", "rel", "rela", "cui2", "cui2_uri", "cui2_name", "sab", "relation_uri"])
    _write_csv(data_dir / "umls_semantic_types.csv", semtype_rows,
               ["cui", "cui_uri", "tui", "semantic_type_name"])

    if failed:
        log.warning("%d CUIs failed and were skipped: %s", len(failed), ", ".join(failed))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--version", default="current",
                         help="UMLS release version for the UTS API (default: current)")
    parser.add_argument("--rate-limit-delay", type=float, default=0.05,
                         help="Seconds to sleep between API calls (default: 0.05)")
    parser.add_argument("--refresh", action="store_true",
                         help="Ignore the on-disk cache and refetch every CUI")
    args = parser.parse_args()

    cache_dir = args.cache_dir or (args.data_dir / "umls_cache")
    build(args.data_dir, cache_dir, args.version, args.rate_limit_delay, args.refresh)


if __name__ == "__main__":
    main()

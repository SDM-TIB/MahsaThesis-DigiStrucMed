"""Search Wikidata and export a condition and its treatments in English/German.

Install: python -m pip install requests
Run:     python wikidata_medical.py "arrhythmias" -o arrhythmias.json
API:     https://doc.wikimedia.org/Wikibase/master/js/rest-api/
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API = "https://www.wikidata.org/w/rest.php/wikibase/v1"
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


class Wikidata:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "GuideLineKG-MedicalExport/1.0 (Wikidata research script)",
            "Accept": "application/json",
        })
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )))
        self.cache = {}

    def get(self, path, **params):
        response = self.session.get(f"{API}{path}", params=params, timeout=60)
        response.raise_for_status()
        return response.json()

    def item(self, item_id):
        if item_id not in self.cache:
            self.cache[item_id] = self.get(f"/entities/items/{item_id}")
        return self.cache[item_id]

    def names(self, item_id):
        item = self.item(item_id)
        return {
            "id": item["id"],
            "url": f"https://www.wikidata.org/wiki/{item['id']}",
            "labels": {lang: item.get("labels", {}).get(lang) for lang in LANGUAGES},
            "aliases": {lang: item.get("aliases", {}).get(lang, []) for lang in LANGUAGES},
        }

    def values(self, item_id, property_id):
        """Return distinct explicit values, excluding deprecated statements."""
        values = []
        for statement in self.item(item_id).get("statements", {}).get(property_id, []):
            value = statement["value"]
            if statement["rank"] != "deprecated" and value["type"] == "value":
                if value["content"] not in values:
                    values.append(value["content"])
        return values

    def related(self, item_id, property_id):
        return [self.names(qid) for qid in self.values(item_id, property_id)]

    def export(self, term, language="en"):
        results = self.get("/search/items", q=term, language=language, limit=1)["results"]
        if not results:
            raise ValueError(f"No Wikidata item found for {term!r}.")
        first = results[0]
        item_id = first["id"]
        print(f"Selected first result: {item_id}", flush=True)
        condition = self.names(item_id)
        for name in ("instance_of", "subclass_of", "has_effect", "studied_by"):
            condition[name] = self.related(item_id, PROPERTIES[name])

        treatments = []
        treatment_ids = self.values(item_id, "P2176")
        for index, treatment_id in enumerate(treatment_ids, 1):
            print(f"Fetching treatment {index}/{len(treatment_ids)}: {treatment_id}", flush=True)
            treatment = self.names(treatment_id)
            treatment["medical_condition_treated"] = self.related(treatment_id, "P2175")
            treatment["drugbank_id"] = self.values(treatment_id, "P715")
            treatments.append(treatment)
        condition["drug_or_therapy_used_for_treatment"] = treatments
        return {
            "query": term,
            "search_language": language,
            "selected_search_result": first,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "api": API,
            "properties": PROPERTIES,
            "scope": "Direct explicit non-deprecated statements only; no recursive or inverse inference. Missing labels are null; missing lists are empty. Unknown/no-value statements are omitted.",
            "condition": condition,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("term", nargs="?", default="arrhythmias")
    parser.add_argument("-o", "--output", type=Path, default=Path("arrhythmias.json"))
    parser.add_argument("--language", choices=LANGUAGES, default="en", help="Search language")
    args = parser.parse_args()
    client = Wikidata()
    try:
        data = client.export(args.term, args.language)
        args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (requests.RequestException, ValueError, OSError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    finally:
        client.session.close()
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()

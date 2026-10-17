"""Step 2 of the Wikidata-sourcing pipeline (see ../README.md): one row
per distinct surface form extracted by the stage-4 pipeline, with its
entity type(s) and an ordered list of search_candidates - the original text
plus every cheap rewrite that might find a match Wikidata search on the raw
text would miss.

Why this exists: Wikidata has no entry for a phrase like "pregnant patient
with sustained VA" - Wikidata models atomic concepts, not descriptive
population sentences, and "VA" inside it is also an abbreviation a plain
lexical search cannot resolve. This script does not (cannot) make Wikidata
have an item for the whole sentence; it derives shorter, more atomic
candidate phrases from it, most-specific first:

0. Skipped entirely if the surface form itself is over MAX_SURFACE_FORM_LENGTH
   chars - see that constant's comment. These are whole paragraphs of
   clinical instructions mis-captured as a single entity span by the stage-4
   extraction, not search terms at any length (confirmed against
   level4-dataset-full: 128 of 17,780 entities, 0.7%).
1. The original surface form.
2. Abbreviation-expanded ("...sustained VA" -> "...sustained ventricular
   arrhythmia"), via ../01-abbreviation-extraction/output/merged_abbreviations.json (built by
   extract_abbreviations.py / merge_abbreviations.py).
3. Connector-split: break the text on standalone "with"/"and"/"or" into its
   clauses (e.g. "pregnant patient with sustained VA" -> "pregnant patient",
   "sustained VA") - each clause is itself also abbreviation-expanded.

This is a data-preprocessing step only: it produces candidates, it does not
search Wikidata or decide which candidate to use - that is step 3,
search_wikidata_candidates.py. It also never edits the raw extraction JSON
files themselves - length filtering only affects what becomes a search
candidate here, not the source-of-truth data step 3's own graph is built
from.

Matching/splitting details:
- Abbreviation matching is whole-word and case-sensitive (VA must not match
  inside "lava"), longest-token-first (so "VA-ECMO" wins over "VA" when both
  are present), and covers recorded plural forms (VAs -> expansion + "s", a
  naive pluralization heuristic).
- Connector-splitting is a blunt heuristic, not a parse: "black and white"
  would split too, for example. It is meant to raise the odds of an
  eventual exact/fuzzy match, not to produce grammatically certain phrases -
  a bad candidate that fails to match Wikidata is harmless; it just means
  wasted tiers, not a wrong result.
- The abbreviation dictionary is global (merged across every document
  extract_abbreviations.py has processed), not scoped to the one guideline a
  given surface form came from - there is no crosswalk yet between a
  stage-4 file's document_sha256 and a 01-abbreviation-extraction/data/input_md/ folder name.

Usage (run from 02-wikidata-enrichment/):
    python src/prepare_wikidata_search.py ../03-json-knowledge-graph/data/level4-dataset --abbreviations ../01-abbreviation-extraction/output/merged_abbreviations.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CONNECTOR_PATTERN = re.compile(r"\s+(with|and|or)\s+", re.IGNORECASE)
MIN_CANDIDATE_LENGTH = 2

# Empirical cutoff, not arbitrary: across every file in level4-dataset-full
# (17,780 entities), surface forms up to ~140 chars are still legitimate
# clause-level POPULATION/CONDITION_OR_FINDING/etc. qualifiers by this
# schema's own design (e.g. "patients with ischemic heart disease and ICD
# shocks for sustained monomorphic VT..."), and split_on_connectors() above
# already exists to shorten those for search. Past ~150 chars every sampled
# case (128 of 17,780, 0.7%) was instead a whole paragraph of clinical
# instructions mis-captured as a single entity span (up to 752 chars) - not
# a real search term at any length, and the original trigger for a live
# HTTP 431 from search_wikidata_candidates.py's batched SPARQL query
# (see search_wikidata_candidates.py's run_batch docstring).
MAX_SURFACE_FORM_LENGTH = 150

# entity_type values that, per this schema's own definitions
# (03-json-knowledge-graph/ontology/ds-ontology-tables.ttl: "ValueEntity:
# a bare numeric/qualitative value span", "DoseEntity: a medication dose
# span (e.g. '81 mg')", similarly StrengthEntity), are measurements, not
# named concepts - "<300 m", "650 mg", "n = 1276" will never have a
# Wikidata item, matching or not. A surface form is skipped only if EVERY
# occurrence of it across the whole dataset was one of these types (a
# surface form also seen as e.g. CONDITION_OR_FINDING elsewhere is kept,
# since it may have real conceptual meaning in that other context).
NON_CONCEPT_ENTITY_TYPES = {"VALUE", "DOSE", "STRENGTH"}


def load_abbreviations(path: Path) -> dict[str, str]:
    """Return {token: expansion} for every abbreviation and its recorded plural forms."""
    data = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    for entry in data.values():
        lookup[entry["abbreviation"]] = entry["expansion"]
        for plural in entry.get("plural_forms_seen", []):
            lookup[plural] = entry["expansion"] + "s"
    return lookup


def build_expander(lookup: dict[str, str]):
    """One compiled regex matching any known abbreviation as a whole word,
    longest token first so e.g. "VA-ECMO" wins over "VA" when both match."""
    if not lookup:
        return lambda text: None
    tokens = sorted(lookup, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in tokens) + r")\b")

    def expand(text: str) -> str | None:
        if not pattern.search(text):
            return None
        return pattern.sub(lambda m: lookup[m.group(1)], text)

    return expand


def split_on_connectors(text: str) -> list[tuple[str, str | None]]:
    """Break on standalone with/and/or; only meaningful with >= 2 clauses.

    Returns (piece, connector) pairs, where connector is the word that
    joined this piece to the PREVIOUS one (None for the first piece) - e.g.
    "A and B or C" -> [("A", None), ("B", "and"), ("C", "or")]. The old
    version threw the connector word away entirely (a non-capturing
    regex), so "A and B" and "A or B" both collapsed into the identical
    ["A", "B"] with no way for a caller to tell a conjunction from an
    alternative apart. CONNECTOR_PATTERN now captures the connector, so
    re.split() returns [piece, connector, piece, connector, piece, ...]
    (always odd length, connector at every odd index) - the two are then
    zipped back together here instead of being lost."""
    parts = CONNECTOR_PATTERN.split(text)
    pieces = []
    for index in range(0, len(parts), 2):
        piece = parts[index].strip()
        if not piece:
            continue
        connector = parts[index - 1].lower() if index > 0 else None
        pieces.append((piece, connector))
    return pieces if len(pieces) > 1 else []


def build_search_candidates(text: str, expand) -> list[dict]:
    """Ordered, deduplicated candidate list, each tagged with how it was
    derived so a later search step can weigh or report on them differently.

    Whenever a piece of text contains a known abbreviation, its expanded
    form is listed BEFORE the bare form, not after. A bare abbreviation like
    "ICD" or "RM" is prone to an exact-match search landing on some
    unrelated Wikidata item that happens to share that label/alias (a
    disease-classification system, an aircraft model) rather than the
    intended clinical meaning - confirmed against live Wikidata, not
    hypothetical. Putting the disambiguated form first means a search step
    that takes "the first candidate that matches" tries the trustworthy
    version before the collision-prone one.

    Connector-split candidates carry "connector" (the with/and/or that
    joined this piece to the PREVIOUS one - see split_on_connectors) and
    "split_index" (its position among the split pieces), so "A and B" and
    "A or B" stay distinguishable downstream instead of both collapsing
    into an identical, connector-less ["A", "B"].

    Every piece is guaranteed at least one candidate, even a bare 1-char
    one: MIN_CANDIDATE_LENGTH exists to stop a noisy short token being
    searched WHEN a better alternative exists, not to make a piece
    disappear from search entirely when it has no better alternative.
    Without this, a real single-letter abbreviation with no dictionary
    entry (the abbreviation dict does have 1-char entries, e.g. "R" ->
    "randomized", "F" -> "feces") produced zero candidates for its row -
    not deprioritized, just silently never searched."""
    candidates: list[dict[str, str | int]] = []
    seen = set()

    def add(candidate: str | None, derivation: str, connector: str | None = None,
            split_index: int | None = None, min_length: int = MIN_CANDIDATE_LENGTH) -> bool:
        # Re-check MAX_SURFACE_FORM_LENGTH here, not just on the raw surface
        # form at the collect_entities() call site: abbreviation expansion
        # can push an under-the-limit surface form back over it (several
        # short abbreviations in one clause each expanding to a long term).
        if not (candidate and min_length <= len(candidate) <= MAX_SURFACE_FORM_LENGTH
                and candidate not in seen):
            return False
        seen.add(candidate)
        entry: dict[str, str | int] = {"text": candidate, "derivation": derivation}
        if connector is not None:
            entry["connector"] = connector
        if split_index is not None:
            entry["split_index"] = split_index
        candidates.append(entry)
        return True

    def add_with_expansion_first(raw_text: str, raw_derivation: str, expanded_derivation: str,
                                  connector: str | None = None, split_index: int | None = None) -> None:
        added_expanded = add(expand(raw_text), expanded_derivation, connector, split_index)
        added_raw = add(raw_text, raw_derivation, connector, split_index)
        if not added_expanded and not added_raw:
            # Neither form qualified (typically: no known expansion, and the
            # bare text is under MIN_CANDIDATE_LENGTH) - fall back to the
            # bare text anyway rather than leaving this piece with nothing
            # to search. add() already skips this if raw_text turned out to
            # be a duplicate of something already added.
            add(raw_text, "short_form_fallback", connector, split_index, min_length=1)

    add_with_expansion_first(text, "surface_form", "abbreviation_expanded")
    for split_index, (piece, connector) in enumerate(split_on_connectors(text)):
        add_with_expansion_first(piece, "connector_split", "connector_split_abbreviation_expanded",
                                  connector=connector, split_index=split_index)

    # Last-resort case-folded fallback, tried only after every original-case
    # candidate above: Wikidata's canonical label for many common-noun
    # clinical/drug concepts is lowercase, while an extracted span keeps
    # whatever capitalization its source sentence gave it (e.g.
    # sentence-initial "Haemodialysis"). Confirmed live: "Haemodialysis"
    # finds nothing, "haemodialysis" matches; same for "Thiazolidinediones"
    # vs "thiazolidinediones". Tried last, same reasoning as putting the
    # abbreviation-expanded form before the bare abbreviation above -
    # lowercasing a short token risks a wrong collision, so only fall back
    # to it once every more-trustworthy candidate has already missed.
    for candidate in list(candidates):
        add(str(candidate["text"]).lower(), "lowercased")

    return candidates


def collect_entities(dataset_dir: Path) -> tuple[dict[str, set[str]], int]:
    """{surface_form: {entity_type, ...}} across every entity in every file,
    plus a count of surface forms skipped for exceeding
    MAX_SURFACE_FORM_LENGTH (see its comment - these are paragraph-level
    extraction spans, not search terms, at any length)."""
    entities: dict[str, set[str]] = {}
    skipped_too_long = 0
    for file in sorted(dataset_dir.glob("*.json")):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for text_unit in data.get("text_units", []):
            for entity in text_unit.get("entities", []):
                surface_form = entity.get("surface_form")
                if not surface_form:
                    continue
                if len(surface_form) > MAX_SURFACE_FORM_LENGTH:
                    skipped_too_long += 1
                    continue
                types = entities.setdefault(surface_form, set())
                entity_type = entity.get("entity_type")
                if entity_type:
                    types.add(entity_type)
    return entities, skipped_too_long


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="Directory of stage-4 extraction JSON files (level4-dataset/)")
    parser.add_argument("--abbreviations", type=Path, required=True,
                         help="merged_abbreviations.json from the 01-abbreviation-extraction/ pipeline")
    parser.add_argument("-o", "--output", type=Path, default=Path("wikidata_search/wikidata_search_input.json"))
    args = parser.parse_args()

    entities, skipped_too_long = collect_entities(args.dataset_dir)
    expand = build_expander(load_abbreviations(args.abbreviations))

    rows = []
    multi_candidate_count = 0
    skipped_non_concept = 0
    for surface_form in sorted(entities):
        types = entities[surface_form]
        if types and types <= NON_CONCEPT_ENTITY_TYPES:
            skipped_non_concept += 1
            continue
        candidates = build_search_candidates(surface_form, expand)
        if len(candidates) > 1:
            multi_candidate_count += 1
        rows.append({
            "surface_form": surface_form,
            "entity_types": sorted(types),
            "search_candidates": candidates,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} distinct surface form(s), {multi_candidate_count} with additional search candidate(s) -> {args.output}")
    if skipped_too_long:
        print(f"Skipped {skipped_too_long} entity occurrence(s) with a surface_form over "
              f"{MAX_SURFACE_FORM_LENGTH} chars (paragraph-level extraction spans, not search terms).")
    if skipped_non_concept:
        print(f"Skipped {skipped_non_concept} surface form(s) only ever seen typed "
              f"{sorted(NON_CONCEPT_ENTITY_TYPES)} (measurements, not searchable concepts).")


if __name__ == "__main__":
    main()

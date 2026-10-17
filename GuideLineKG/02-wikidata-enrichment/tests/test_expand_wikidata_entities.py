"""Regression checks for alias cleanup in expand_wikidata_entities.py."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import expand_wikidata_entities as x  # noqa: E402


class CleanAliasesTest(unittest.TestCase):
    def test_drops_case_only_duplicates_folding_to_lowercase(self):
        # Output is the folded form, not either original casing - it has to
        # be, since that's what the IRI rr:template builds from ends up as.
        aliases = {"Dyclocaine", "dyclocaine", "Diclonina"}
        self.assertEqual(x._clean_aliases(aliases, "dyclonine"),
                          ["diclonina", "dyclocaine"])

    def test_drops_alias_matching_label_in_any_casing(self):
        aliases = {"Dyclonine", "dyclonine", "Dyclocaine"}
        self.assertEqual(x._clean_aliases(aliases, "dyclonine"), ["dyclocaine"])

    def test_keeps_distinct_aliases_untouched_but_folded(self):
        aliases = {"Dyclocaine", "Dycloninum"}
        self.assertEqual(x._clean_aliases(aliases, "dyclonine"),
                          ["dyclocaine", "dycloninum"])

    def test_deterministic_across_input_order(self):
        a = x._clean_aliases({"Zeta", "zeta", "Alpha"}, "dyclonine")
        b = x._clean_aliases({"Alpha", "zeta", "Zeta"}, "dyclonine")
        self.assertEqual(a, b)
        self.assertEqual(a, ["alpha", "zeta"])

    def test_folds_whitespace_the_same_way_normalize_surface_forms_does(self):
        # Matches 03-json-knowledge-graph/src/normalize_surface_forms.py's
        # _normalized_key exactly (casefold + collapse whitespace), not
        # just casefold - a SurfaceForm node folded from "Heart  Failure"
        # (double space) must land on the same IRI as this does.
        aliases = {"Heart  Failure", "heart failure "}
        self.assertEqual(x._clean_aliases(aliases, "dyclonine"), ["heart failure"])


class MatchedSurfaceFormsTest(unittest.TestCase):
    """The original 'surface_form' text a guideline mention was extracted as
    (e.g. "all individuals with HF") is only related to a Wikidata match's
    own label (e.g. "heart failure") through wikidata_matches.json -
    main() previously reduced every matched row straight to a bare QID set
    and never carried that text any further, so nothing in the final graph
    could tell the two were ever connected. These lock in the fix."""

    def test_end_to_end_output_matches_step_3s_own_surface_form_fold(self):
        """Reproduces a real bug found after the first version of this fix:
        finalize_group() output the ORIGINAL casing ("all individuals with
        HF"), so mapping-wikidata-entities.rml.ttl minted
        ds:matchedSurfaceForm -> entity/all%20individuals%20with%20HF - a
        different, empty IRI from entity/all%20individuals%20with%20hf,
        the real node 03-json-knowledge-graph/src/normalize_surface_forms.py
        folds a matching extracted mention onto. The edge didn't error, it
        just pointed at nothing. This checks the two pipelines' outputs
        agree byte-for-byte on the folded text, not just structurally."""
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                                 / "03-json-knowledge-graph" / "src"))
        from normalize_surface_forms import _normalized_key as step3_key  # noqa: E402

        group = {
            "label": "heart failure", "description": "", "also_known_as": set(),
            "wikidata_ids": {"Q181754"}, "hop": 0, "has_data": True,
            "discovered_via": {"match"},
            "matched_surface_forms": {"all individuals with HF"},
        }
        for name in x.ITEM_PROPERTIES:
            group[name] = set()
        for name in x.STRING_PROPERTIES:
            group[name + "s"] = set()
        entry = x.finalize_group("heart failure", group)

        self.assertEqual(entry["matched_surface_forms"],
                          [step3_key("all individuals with HF")])

    def test_merge_into_groups_threads_surface_forms_for_hop_0_only(self):
        qid_to_label = {"Q181754": "heart failure"}
        source = {"Q181754": {"also_known_as": set(), "description": "", "targets": set()}}
        groups: dict = {}
        x.merge_into_groups(groups, qid_to_label, ["Q181754"], source, 0, {}, has_data=True,
                             surface_forms={"Q181754": {"all individuals with HF"}})
        self.assertEqual(groups["heart failure"]["matched_surface_forms"],
                          {"all individuals with HF"})

    def test_merge_into_groups_ignores_surface_forms_for_non_hop_0_calls(self):
        # Hop-1/2 items were reached by property traversal, not by a search
        # match - passing no surface_forms (the default) must not error and
        # must leave the field empty.
        qid_to_label = {"Q1": "some item"}
        source = {"Q1": {"also_known_as": set(), "description": "", "targets": set()}}
        groups: dict = {}
        x.merge_into_groups(groups, qid_to_label, ["Q1"], source, 1, {}, has_data=True)
        self.assertEqual(groups["some item"]["matched_surface_forms"], set())

    def test_finalize_group_cleans_and_exposes_matched_surface_forms(self):
        group = {
            "label": "heart failure", "description": "", "also_known_as": set(),
            "wikidata_ids": {"Q181754"}, "hop": 0, "has_data": True,
            "discovered_via": {"match"},
            "matched_surface_forms": {"all individuals with HF", "Heart Failure"},
        }
        for name in x.ITEM_PROPERTIES:
            group[name] = set()
        for name in x.STRING_PROPERTIES:
            group[name + "s"] = set()
        entry = x.finalize_group("heart failure", group)
        # "Heart Failure" is dropped (matches the label case-insensitively,
        # same rule _clean_aliases already applies to also_known_as), and
        # the survivor is folded ("all individuals with hf") - not the
        # original casing - so it lands on the same IRI step 3's own
        # normalize_surface_forms.py would fold a matching mention to.
        self.assertEqual(entry["matched_surface_forms"], ["all individuals with hf"])


if __name__ == "__main__":
    unittest.main()

"""Regression checks for cross-graph surface-form canonicalization."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rdflib import Graph, Namespace, RDF, RDFS, Literal  # noqa: E402
import merge_graphs as m  # noqa: E402

DS = Namespace("http://digistrucmed.org/")


def _surface_form(graph, text, extra_types=()):
    node = DS["entity/" + text.replace(" ", "%20").replace("-", "%2D")]
    graph.add((node, RDF.type, DS.SurfaceForm))
    for t in extra_types:
        graph.add((node, RDF.type, t))
    graph.add((node, RDFS.label, Literal(text)))
    return node


class MergeGraphsTest(unittest.TestCase):
    def test_merges_hyphen_space_and_plural_variants(self):
        graph = Graph()
        _surface_form(graph, "beta-blocker")
        _surface_form(graph, "beta-blockers")
        _surface_form(graph, "beta blockers")

        merged, report = m.canonicalize(graph)

        # None of the raw spellings is itself the casefolded/hyphen-folded
        # canonical form ("beta blocker"), so all three remap onto a newly
        # synthesized canonical node - same "every variant moves, including
        # the one that looks closest to canonical" behavior
        # normalize_surface_forms.py already has (see its own
        # test_idempotent_on_second_run: "Heart"/"HEART" -> both move to "heart").
        self.assertEqual(merged, 3)
        self.assertEqual(len(report), 1)
        self.assertCountEqual(report[0]["merged_surface_forms"],
                               ["beta blockers", "beta-blocker", "beta-blockers"])

    def test_does_not_merge_qualifier_phrase(self):
        graph = Graph()
        _surface_form(graph, "beta blocker")
        _surface_form(graph, "evidence-based beta blockers")

        merged, report = m.canonicalize(graph)

        self.assertEqual(merged, 0)
        self.assertEqual(report, [])

    def test_trusts_hasPluralForm_for_abbreviation_plurals(self):
        graph = Graph()
        va = _surface_form(graph, "VA", extra_types=[DS.Abbreviation])
        vas = _surface_form(graph, "VAs", extra_types=[DS.Abbreviation])
        graph.add((va, DS.hasPluralForm, vas))

        merged, report = m.canonicalize(graph)

        self.assertEqual(merged, 2)  # canonical is the casefolded "va", so both remap
        self.assertCountEqual(report[0]["merged_surface_forms"], ["VA", "VAs"])

    def test_does_not_blindly_merge_abbreviation_lookalike_plural(self):
        """RAA (right atrial appendage) and RAAS (renin-angiotensin-aldosterone
        *system*) merely share a text prefix - RAAS is not the plural of RAA.
        Unlike plain-English nouns, abbreviations only merge via an explicit
        ds:hasPluralForm edge (see test above), never by guessing by suffix."""
        graph = Graph()
        _surface_form(graph, "RAA", extra_types=[DS.Abbreviation])
        _surface_form(graph, "RAAS", extra_types=[DS.Abbreviation])

        merged, report = m.canonicalize(graph)

        self.assertEqual(merged, 0)
        self.assertEqual(report, [])

    def test_closed_world_plural_fold_for_plain_terms(self):
        graph = Graph()
        _surface_form(graph, "diuretic")
        _surface_form(graph, "diuretics")
        _surface_form(graph, "status")  # no "statu" node -> must stay alone

        merged, report = m.canonicalize(graph)

        self.assertEqual(merged, 1)
        self.assertEqual(len(report), 1)
        self.assertCountEqual(report[0]["merged_surface_forms"], ["diuretic", "diuretics"])


if __name__ == "__main__":
    unittest.main()

"""Regression checks for merging case/whitespace variants of surface forms."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rdflib import Graph, Namespace, RDF, RDFS, Literal, URIRef  # noqa: E402
import normalize_surface_forms as n  # noqa: E402

DS = Namespace("http://digistrucmed.org/")


class NormalizeSurfaceFormsTest(unittest.TestCase):
    def test_merges_case_and_whitespace_variants_across_documents(self):
        graph = Graph()
        lower = DS["entity/beta%20blocker"]
        upper = DS["entity/Beta%20%20Blocker"]
        mention_a = DS["A/e_0001"]
        mention_b = DS["B/e_0001"]
        for node, label in [(lower, "beta blocker"), (upper, "Beta  Blocker")]:
            graph.add((node, RDF.type, DS.SurfaceForm))
            graph.add((node, RDFS.label, Literal(label)))
        graph.add((mention_a, DS.surfaceForm, lower))
        graph.add((mention_b, DS.surfaceForm, upper))

        merged = n.normalize_surface_forms(graph)

        canonical = DS["entity/beta%20blocker"]
        self.assertEqual(merged, 1)
        self.assertEqual(set(graph.objects(mention_a, DS.surfaceForm)), {canonical})
        self.assertEqual(set(graph.objects(mention_b, DS.surfaceForm)), {canonical})
        self.assertEqual(set(graph.objects(canonical, RDFS.label)),
                          {Literal("beta blocker"), Literal("Beta  Blocker")})
        self.assertNotIn(upper, set(graph.subjects(RDF.type, DS.SurfaceForm)))

    def test_noop_when_already_canonical(self):
        graph = Graph()
        node = DS["entity/beta%20blocker"]
        graph.add((node, RDF.type, DS.SurfaceForm))
        graph.add((node, RDFS.label, Literal("beta blocker")))
        before = set(graph)

        merged = n.normalize_surface_forms(graph)

        self.assertEqual(merged, 0)
        self.assertEqual(set(graph), before)

    def test_idempotent_on_second_run(self):
        graph = Graph()
        graph.add((DS["entity/Heart"], RDF.type, DS.SurfaceForm))
        graph.add((DS["entity/HEART"], RDF.type, DS.SurfaceForm))

        first = n.normalize_surface_forms(graph)
        second = n.normalize_surface_forms(graph)

        self.assertEqual(first, 2)  # Both "Heart" and "HEART" move to canonical "heart".
        self.assertEqual(second, 0)


if __name__ == "__main__":
    unittest.main()

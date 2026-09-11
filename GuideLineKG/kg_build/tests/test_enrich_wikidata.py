"""Offline regression checks for condition -> treatment -> drug-detail expansion."""
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import enrich_wikidata as e


class TreatmentExpansionTest(unittest.TestCase):
    def test_forward_reverse_drugbank_and_shared_cache(self):
        base = [(e.WD.Q1, e.WDT.P2176, e.WD.Q2)]
        drug = [(e.WD.Q2, e.WDT.P715, e.Literal("DB00001")),
                (e.WD.Q2, e.WDT.P267, e.Literal("A01AA01")),
                (e.WD.Q2, e.WDT.P769, e.WD.Q99)]
        reverse = [(e.WD.Q3, e.WDT.P2175, e.WD.Q1),
                   (e.WD.Q3, e.WDT.P715, e.Literal("DB00002")),
                   (e.WD.Q3, e.URIRef("http://schema.org/description"), e.Literal("Beispiel", lang="de"))]
        cache = {}
        with patch.object(e, "search_treatments_api", return_value=(["Q2", "Q3", "Q4"], False)), patch.object(e, "fetch_with_api", side_effect=lambda _, q: {"Q2": drug, "Q3": reverse, "Q4": []}[q]) as fetch:
            extra, report = e.fetch_related_treatments(Mock(), "Q1", base, cache)
            self.assertEqual(set(report["expanded_items"]), {"Q2", "Q3"})
            self.assertEqual(report["expanded_items"]["Q2"]["drugbank_ids"], ["DB00001"])
            e.fetch_related_treatments(Mock(), "Q1", base, cache)
            self.assertEqual(fetch.call_count, 3)  # Includes unconfirmed reverse result Q4.
            self.assertNotIn("Q99", cache)  # Interactions are not recursively expanded.
        graph = e.Graph()
        surface = e.URIRef(str(e.DS) + "entity/condition")
        graph.add((e.DS.mention, e.DS.surfaceForm, surface))
        original = set(graph)
        e.add_to_knowledge_graph(graph, {surface}, {"id": "Q1", "similarity": 1.0}, base + extra)
        self.assertTrue(original.issubset(graph))
        drug_node = e.local_iri(str(e.WD.Q2))
        self.assertIn((drug_node, e.DS.drugbank_id, e.Literal("DB00001")), graph)
        self.assertNotIn((e.local_iri(str(e.WD.Q1)), e.DS.drugbank_id, e.Literal("DB00001")), graph)
        self.assertEqual(len(e.Graph().parse(data=graph.serialize(format="nt"), format="nt")), len(graph))

    def test_partial_failure_retains_successful_treatment(self):
        base = [(e.WD.Q1, e.WDT.P2176, e.WD.Q2), (e.WD.Q1, e.WDT.P2176, e.WD.Q3)]
        def fetch(_, qid):
            if qid == "Q3":
                raise e.requests.Timeout("drug unavailable")
            return [(e.WD.Q2, e.RDFS.label, e.Literal("procedure", lang="en"))]
        with patch.object(e, "search_treatments_api", side_effect=e.requests.Timeout()), patch.object(e, "fetch_with_api", side_effect=fetch):
            triples, report = e.fetch_related_treatments(Mock(), "Q1", base, {})
        self.assertTrue(triples)
        self.assertIn("Q3", report["failed_items"])
        self.assertIsNotNone(report["reverse_search_error"])
        self.assertEqual(report["expanded_items"]["Q2"]["drugbank_ids"], [])

    def test_search_pagination_and_limit(self):
        session = Mock()
        session.get.return_value.json.side_effect = [
            {"query": {"search": [{"title": "Q2"}]}, "continue": {"sroffset": 1}},
            {"query": {"search": [{"title": "Q3"}]}}]
        ids, truncated = e.search_treatments_api(session, "Q1", 2)
        self.assertEqual(ids, ["Q2", "Q3"])
        self.assertFalse(truncated)
        self.assertEqual(session.get.call_args.kwargs["params"]["srsearch"], "haswbstatement:P2175=Q1")
        self.assertEqual(session.get.call_args.kwargs["params"]["sroffset"], 1)

    def test_drug_class_members_have_ids_on_drug_not_class(self):
        drug = [(e.WD.Q2, e.WDT.P279, e.WD.Q1),
                (e.WD.Q2, e.WDT.P715, e.Literal("DB00001"))]
        with patch.object(e, "search_treatments_api", side_effect=[([], False), (["Q2"], False)]), patch.object(e, "fetch_with_api", return_value=drug):
            extra, report = e.fetch_related_treatments(Mock(), "Q1", [], {})
        self.assertEqual(report["expanded_items"]["Q2"]["found_by"], "drug class member: P279")
        graph = e.Graph()
        e.add_to_knowledge_graph(graph, {e.DS["term"]}, {"id": "Q1", "similarity": 1}, extra)
        self.assertIn((e.local_iri(str(e.WD.Q2)), e.DS.drugbank_id, e.Literal("DB00001")), graph)
        self.assertFalse(list(graph.objects(e.local_iri(str(e.WD.Q1)), e.DS.drugbank_id)))

    def test_name_search_excludes_papers(self):
        session = Mock()
        session.get.return_value.json.return_value = {"search": [
            {"id": "Q1", "label": "VT ablation", "description": "wissenschaftlicher Artikel"}]}
        self.assertIsNone(e.search_wikidata_api(session, "VT ablation"))

    def test_names_ids_and_drug_type(self):
        graph = e.Graph()
        triples = [(e.WD.Q412515, e.RDFS.label, e.Literal("bisoprolol", lang="en")),
                   (e.WD.Q412515, e.RDFS.label, e.Literal("Bisoprolol", lang="de")),
                   (e.WD.Q412515, e.WDT.P715, e.Literal("DB00612"))]
        e.add_to_knowledge_graph(graph, {e.DS["entity/beta%20blocker"]}, {"id": "Q1", "similarity": 1}, triples)
        drug = e.local_iri(str(e.WD.Q412515))
        self.assertIn((drug, e.RDF.type, e.DS.Drug), graph)
        self.assertIn((drug, e.DS.name, e.Literal("bisoprolol")), graph)
        self.assertIn((drug, e.DS.wikidataId, e.Literal("Q412515")), graph)
        self.assertIn((drug, e.DS.wikidataSource, e.WD.Q412515), graph)
        self.assertEqual({v.language for v in graph.objects(drug, e.RDFS.label)}, {"en", "de"})
        self.assertNotIn((drug, e.RDF.type, e.DS.ExtractedEntity), graph)

    def test_clinical_types_reused_without_typing_unrelated_nodes(self):
        for code, expected in [("MEDICATION", e.DS.Drug), ("POPULATION", e.DS.Population),
                               ("ADVERSE_EVENT", e.DS.SideEffect)]:
            graph = e.Graph()
            node = e.DS["entity/example"]
            graph.add((e.DS.mention, e.DS.surfaceForm, node))
            graph.add((e.DS.mention, e.DS.entityType, e.DS["entity-type/" + code]))
            triples = [(e.WD.Q1, e.WDT.P2579, e.WD.Q2)]
            e.add_to_knowledge_graph(graph, {node}, {"id": "Q1", "similarity": 1}, triples)
            item = e.local_iri(str(e.WD.Q1))
            self.assertIn((node, e.RDF.type, expected), graph)
            self.assertIn((item, e.RDF.type, expected), graph)
            self.assertIn((item, e.DS.classificationSource, node), graph)
            self.assertNotIn((e.local_iri(str(e.WD.Q2)), e.RDF.type, expected), graph)

    def test_navigation_keeps_two_document_grades_separate(self):
        graph = e.Graph()
        concept = e.DS["entity/beta%20blocker"]
        item = e.DS["wikidata/entity/Q816759"]
        graph.add((concept, e.DS.wikidataMatch, item))
        for document, value in [("A", "I"), ("B", "IIa")]:
            mention = e.DS[document + "/e_0001"]
            assertion = e.DS[document + "/na_0001"]
            record = e.DS[document + "/gr_0001"]
            graph.add((mention, e.DS.surfaceForm, concept))
            graph.add((assertion, e.DS.hasTargetEntity, mention))
            graph.add((assertion, e.DS.hasSourceGradeRecord, record))
            graph.add((record, e.DS.hasGradeValue, e.DS["ClassOfRecommendation_" + value]))
        original = set(graph)
        e.connect_graph_navigation(graph)
        self.assertTrue(original.issubset(graph))
        self.assertIn((item, e.DS.matchedJsonConcept, concept), graph)
        for document in ["A", "B"]:
            mention = e.DS[document + "/e_0001"]
            self.assertEqual(set(graph.objects(mention, e.DS.occursInAssertion)), {e.DS[document + "/na_0001"]})
        self.assertEqual(len(list(graph.objects(concept, e.DS.hasSourceMention))), 2)
        self.assertFalse(list(graph.objects(item, e.DS.hasClassOfRecommendation)))
        self.assertEqual(e.connect_graph_navigation(graph), 0)


if __name__ == "__main__":
    unittest.main()


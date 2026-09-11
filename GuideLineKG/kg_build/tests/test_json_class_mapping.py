"""Run: py -3.12 -m unittest discover -s kg_build/tests -p test_json_class_mapping.py"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from rdflib import Graph, Namespace, RDF, OWL, Literal, RDFS

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from enrich_json_types import ENTITY_TYPE_TO_CLASS, GRADE_DIMENSION_TO_CLASS, enrich

DS = Namespace("http://digistrucmed.org/")


class MappingClassesTest(unittest.TestCase):
    def test_raw_rdfizer_types(self):
        lookup = json.loads((BASE / "mappings/ontology-class-lookup.json").read_text())
        ontology = Graph().parse(BASE / "ontology/ds-ontology-tables.ttl")
        for row in lookup["entity_types"] + lookup["entity_domain_types"] + lookup["grade_dimensions"]:
            from rdflib import URIRef
            self.assertIn((URIRef(row["class_iri"]), RDF.type, OWL.Class), ontology)
        for row in lookup["entity_concept_types"]:
            self.assertIn((URIRef(row["class_iri"]), RDF.type, OWL.Class), ontology)
        data = json.loads((BASE / "mappings/level4.json").read_text(encoding="utf-8"))
        unit = data["text_units"][0]
        codes = list(ENTITY_TYPE_TO_CLASS) + ["ENTITY_OTHER", "UNKNOWN"]
        template = unit["entities"][0]
        unit["entities"] = [dict(template, entity_id=f"e_{i}", entity_type=code, surface_form=f"concept_{code}")
                            for i, code in enumerate(codes)]
        unit["source_grade_records"] = [
            {"grade_record_id": f"g_{i}", "grade_dimension": code, "raw_grade": ["I", "B-NR", "UNRECOGNIZED"][i]}
            for i, code in enumerate([*GRADE_DIMENSION_TO_CLASS, "UNKNOWN"])]
        unit["source_grade_records"].append({"grade_record_id": "g_repeat",
            "grade_dimension": "CLASS_OF_RECOMMENDATION", "raw_grade": "I"})
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            source = work / "input.json"
            source.write_text(json.dumps(data), encoding="utf-8")
            mapping = (BASE / "mappings/mapping-json.rml.ttl").read_text(encoding="utf-8")
            def replace_source(match):
                target = BASE / match[1] if match[1].endswith("ontology-class-lookup.json") else source
                return 'rml:source "' + target.as_posix() + '"'
            mapping = re.sub(r'rml:source "([^"\n]+)"', replace_source, mapping)
            (work / "mapping.ttl").write_text(mapping, encoding="utf-8")
            config = "\n".join([
                "[datasets]", "number_of_datasets: 1", "all_in_one_file: yes",
                "name: test", "remove_duplicate: yes", "enrichment: no",
                "output_folder: " + work.as_posix(), "output_format: n-triples",
                "ordered: yes", "[dataset1]", "name: test",
                "mapping: " + (work / "mapping.ttl").as_posix()])
            (work / "config.ini").write_text(config)
            result = subprocess.run([sys.executable, "-m", "rdfizer", "-c", str(work / "config.ini")],
                env={**os.environ, "PYTHONUTF8": "1"}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            graph = Graph().parse(work / "test.nt", format="nt")
        for i, code in enumerate(codes):
            expected = set(ENTITY_TYPE_TO_CLASS.get(code, ())) | {DS.ExtractedEntity}
            self.assertEqual(set(graph.objects(DS[f"entity-mention/e_{i}"], RDF.type)), expected, code)
        for row in lookup["entity_concept_types"]:
            node = DS["entity/concept_" + row["code"]]
            self.assertIn((node, RDF.type, URIRef(row["class_iri"])), graph)
            self.assertIn((node, DS.name, Literal("concept_" + row["code"])), graph)
        for i in range(3):
            self.assertEqual(set(graph.objects(DS[f"grade-record/g_{i}"], RDF.type)), {DS.SourceGradeRecord})
        self.assertEqual(set(graph.objects(DS["grade-record/g_2"], RDF.type)), {DS.SourceGradeRecord})
        self.assertEqual(set(graph.objects(DS["grade-record/g_0"], DS.hasGradeValue)), {DS.ClassOfRecommendation_I})
        self.assertEqual(set(graph.objects(DS["grade-record/g_1"], DS.hasGradeValue)), {DS["LevelOfEvidence_B-NR"]})
        self.assertFalse(list(graph.objects(DS["grade-record/g_2"], DS.hasGradeValue)))
        self.assertEqual(set(graph.objects(DS["grade-record/g_repeat"], DS.hasGradeValue)), {DS.ClassOfRecommendation_I})
        self.assertEqual(len(list(graph.triples((DS.ClassOfRecommendation_I, RDFS.label, None)))), 1)
        self.assertIn((DS.ClassOfRecommendation_I, RDF.type, DS.ClassOfRecommendation), graph)

    def test_shared_grades_across_documents(self):
        graph = Graph()
        for document in ["doc1", "doc2"]:
            assertion = DS[document + "/assertion"]
            graph.add((assertion, RDF.type, DS.NormativeAssertion))
            for code, value in [("CLASS_OF_RECOMMENDATION", "I"), ("LEVEL_OF_EVIDENCE", "B-NR")]:
                record = DS[document + "/" + code]
                graph.add((record, RDF.type, DS.SourceGradeRecord))
                graph.add((record, DS.gradeDimension, Literal(code)))
                graph.add((record, DS.rawGrade, Literal(value)))
                # Old output incorrectly put this class on the source record.
                graph.add((record, RDF.type, GRADE_DIMENSION_TO_CLASS[code][0]))
                graph.add((assertion, DS.hasSourceGradeRecord, record))
                graph.add((assertion, GRADE_DIMENSION_TO_CLASS[code][1], record))
        enrich(graph)
        self.assertEqual(set(graph.objects(None, DS.hasClassOfRecommendation)), {DS.ClassOfRecommendation_I})
        self.assertEqual(set(graph.objects(None, DS.hasLevelOfEvidence)), {DS["LevelOfEvidence_B-NR"]})
        self.assertEqual(len(list(graph.triples((None, DS.hasSourceGradeRecord, None)))), 4)
        self.assertEqual(len(list(graph.triples((None, DS.hasGradeValue, None)))), 4)
        for record in graph.subjects(RDF.type, DS.SourceGradeRecord):
            self.assertNotIn((record, RDF.type, DS.LevelOfEvidence), graph)
            self.assertNotIn((record, RDF.type, DS.ClassOfRecommendation), graph)
        before = set(graph)
        enrich(graph)
        self.assertEqual(set(graph), before)


    def test_two_documents_keep_local_ids_separate(self):
        from run_json_batch import process_one, merge_all
        template = json.loads((BASE / "mappings/level4.json").read_text(encoding="utf-8"))
        template["text_units"][0]["source_grade_records"] = [{
            "grade_record_id": "gr_0002", "grade_dimension": "LEVEL_OF_EVIDENCE", "raw_grade": "B-NR"}]
        template["text_units"][0]["normative_assertions"][0]["source_grade_record_ids"] = ["gr_0002"]
        combined = Graph()
        mapping_before = (BASE / "mappings/mapping-json.rml.ttl").read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            paths = []
            for document in ["document_A", "document_B"]:
                data = json.loads(json.dumps(template))
                data["semantic_unit_id"] = document
                data["source"]["document_sha256"] = document
                source = work / (document + ".json")
                output = work / (document + ".nt")
                source.write_text(json.dumps(data), encoding="utf-8")
                before = source.read_bytes()
                process_one(source, output)
                self.assertEqual(source.read_bytes(), before)
                paths.append(output)
            output = work / "combined.nt"
            merge_all(paths, output)
            combined.parse(output, format="nt")
        self.assertEqual((BASE / "mappings/mapping-json.rml.ttl").read_bytes(), mapping_before)
        records = set(combined.subjects(RDF.type, DS.SourceGradeRecord))
        self.assertEqual(records, {DS["grade-record/document_A/gr_0002"], DS["grade-record/document_B/gr_0002"]})
        for record in records:
            self.assertEqual(set(combined.objects(record, RDF.type)), {DS.SourceGradeRecord})
            self.assertEqual(set(combined.objects(record, DS.hasGradeValue)), {DS["LevelOfEvidence_B-NR"]})
        for kind, cls in [("entity-mention", DS.ExtractedEntity), ("text-unit", DS.TextUnit),
                          ("normative-assertion", DS.NormativeAssertion), ("grade-record", DS.SourceGradeRecord)]:
            nodes = list(combined.subjects(RDF.type, cls))
            self.assertTrue(any(str(n).startswith(str(DS) + kind + "/document_A/") for n in nodes))
            self.assertTrue(any(str(n).startswith(str(DS) + kind + "/document_B/") for n in nodes))
        self.assertIn((DS["LevelOfEvidence_B-NR"], RDF.type, DS.LevelOfEvidence), combined)
        for assertion, mention in combined.subject_objects(DS.hasTargetEntity):
            self.assertIn((mention, DS.occursInAssertion, assertion), combined)
        for mention, concept in combined.subject_objects(DS.surfaceForm):
            self.assertIn((concept, DS.hasSourceMention, mention), combined)
        for assertion in combined.subjects(RDF.type, DS.NormativeAssertion):
            self.assertEqual(set(combined.objects(assertion, DS.hasLevelOfEvidence)), {DS["LevelOfEvidence_B-NR"]})


if __name__ == "__main__":
    unittest.main()

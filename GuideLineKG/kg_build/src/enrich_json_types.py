"""Complete grade shortcuts and support typing of older JSON-mapping output.

The current mapping assigns entity types and shared grade-value types through
ontology-class-lookup.json joins. This script remains safe to run afterward:
RDF graphs deduplicate the type triples, and this step additionally connects
normative assertions to shared grade IRIs via
hasClassOfRecommendation/hasLevelOfEvidence. Source records link to the same
IRIs through hasGradeValue. Identical dimension/value pairs share one node.
It also supports older output containing only entityType/gradeDimension.

Run: python src/enrich_json_types.py output_json/knowledge_graph_json.nt
Use --output NEW.nt to preserve the original file.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote

from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal, OWL

DS = Namespace("http://digistrucmed.org/")

GRADE_DIMENSION_TO_CLASS = {
    "CLASS_OF_RECOMMENDATION": (DS.ClassOfRecommendation, DS.hasClassOfRecommendation),
    "LEVEL_OF_EVIDENCE": (DS.LevelOfEvidence, DS.hasLevelOfEvidence),
}

# entity_type (the full closed enum from the extraction JSON's entities[]
# schema) -> classes to assert as rdf:type. Every value gets its own
# ds:<Name>Entity class (ontology/ds-ontology-tables.ttl's "entityType
# subclasses" section); where that class is also declared rdfs:subClassOf
# an existing, unrelated-in-name domain class, that class is listed too
# (e.g. MEDICATION -> ds:MedicationEntity AND ds:Drug). Kept in sync by
# hand with the ontology's rdfs:subClassOf declarations - if you add a
# subClassOf there, add the matching second class here too.
ENTITY_TYPE_TO_CLASS = {
    "POPULATION": (DS.PopulationEntity, DS.Population),
    "CONDITION_OR_FINDING": (DS.ConditionOrFindingEntity, DS.PatientCondition),
    "MEDICATION": (DS.MedicationEntity, DS.Drug),
    "PROCEDURE_OR_INTERVENTION": (DS.ProcedureOrInterventionEntity, DS.ClinicalProcedure),
    "TEST_OR_MEASUREMENT": (DS.TestOrMeasurementEntity,),
    "DEVICE": (DS.DeviceEntity, DS.Device),
    "ANATOMY": (DS.AnatomyEntity,),
    "OUTCOME": (DS.OutcomeEntity,),
    "ADVERSE_EVENT": (DS.AdverseEventEntity, DS.SideEffect),
    "VALUE": (DS.ValueEntity,),
    "TIME": (DS.TimeEntity, DS.TemporalConstraint),
    "DOSE": (DS.DoseEntity,),
    "STRENGTH": (DS.StrengthEntity,),
    "ROUTE": (DS.RouteEntity,),
    "FREQUENCY": (DS.FrequencyEntity,),
    "DURATION": (DS.DurationEntity, DS.TemporalConstraint),
    "FORM": (DS.FormEntity,),
    "ACTOR": (DS.ActorEntity,),
    "RATIONALE": (DS.RationaleEntity,),
    # ENTITY_OTHER: deliberately absent - the schema's own catch-all.
}


def enrich(graph: Graph) -> dict[str, int]:
    counts = {"grade_typed": 0, "grade_shortcut_edges": 0, "entity_typed": 0}

    for record in set(graph.subjects(RDF.type, DS.SourceGradeRecord)):
        # Remove legacy category types from occurrences; only values have them.
        graph.remove((record, RDF.type, DS.ClassOfRecommendation))
        graph.remove((record, RDF.type, DS.LevelOfEvidence))
        # Old shortcut ranges would otherwise infer those wrong types again.
        graph.remove((None, DS.hasClassOfRecommendation, record))
        graph.remove((None, DS.hasLevelOfEvidence, record))
        dimension = graph.value(record, DS.gradeDimension)
        mapping = GRADE_DIMENSION_TO_CLASS.get(str(dimension)) if dimension else None
        if mapping is None:
            continue
        grade_class, shortcut_predicate = mapping
        raw_grade = graph.value(record, DS.rawGrade)
        if raw_grade is None or not str(raw_grade).strip():
            continue
        # Global IRI: no document ID, so identical dimension/value pairs merge.
        # Preserve spelling: "1" and "I" are not silently treated as equivalent.
        value = str(raw_grade)
        shared_grade = URIRef(str(grade_class) + "_" + quote(value, safe=""))
        graph.add((shared_grade, RDF.type, OWL.NamedIndividual))
        graph.add((shared_grade, RDF.type, grade_class))
        counts["grade_typed"] += 1
        graph.add((shared_grade, RDFS.label, Literal(value)))
        graph.add((record, DS.hasGradeValue, shared_grade))
        graph.remove((record, RDFS.label, raw_grade))
        for assertion in graph.subjects(DS.hasSourceGradeRecord, record):
            # ds:TextUnit also reaches grade records via ds:hasSourceGradeRecord
            # (see mappings/mapping-json.rml.ttl's TextUnitMap) - only
            # ds:NormativeAssertion subjects get the ds:hasClassOfRecommendation/
            # ds:hasLevelOfEvidence shortcut, since those predicates' domain is
            # ds:Recommendation (ds:NormativeAssertion's superclass).
            if (assertion, RDF.type, DS.NormativeAssertion) not in graph:
                continue
            # Migrate old shortcuts while keeping hasSourceGradeRecord provenance.
            graph.remove((assertion, shortcut_predicate, record))
            if (assertion, shortcut_predicate, shared_grade) not in graph:
                graph.add((assertion, shortcut_predicate, shared_grade))
                counts["grade_shortcut_edges"] += 1

    for entity in set(graph.subjects(RDF.type, DS.ExtractedEntity)):
        entity_type = graph.value(entity, DS.entityType)
        target_classes = ENTITY_TYPE_TO_CLASS.get(str(entity_type).removeprefix("http://digistrucmed.org/entity-type/")) if entity_type else None
        if target_classes is None:
            continue
        for target_class in target_classes:
            graph.add((entity, RDF.type, target_class))
        counts["entity_typed"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="RDFizer n-triples output to enrich")
    parser.add_argument("--output", type=Path, default=None, help="Write here instead of overwriting --path")
    args = parser.parse_args()

    graph = Graph()
    graph.parse(args.path, format="nt")
    before = len(graph)

    counts = enrich(graph)

    out_path = args.output or args.path
    graph.serialize(destination=out_path, format="nt", encoding="utf-8")

    print(
        f"{args.path}: {before} -> {len(graph)} triples "
        f"(+{counts['grade_typed']} shared grade-value types, "
        f"+{counts['grade_shortcut_edges']} hasClassOfRecommendation/hasLevelOfEvidence shortcut edges, "
        f"+{counts['entity_typed']} entity types) -> {out_path}"
    )


if __name__ == "__main__":
    main()

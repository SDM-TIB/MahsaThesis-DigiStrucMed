"""Concept-based prompt enrichment for the questionnaire knowledge graph.

Given a question and the KG entities it mentions (e.g. ``UM17`` or the
composite response id ``217_M010_01``), this module looks up what those
entities *mean* semantically (question category, questionnaire, answer
category, ...) in ``data/qKG/facts.tsv`` and turns that into a short natural
language context block. Prepending that block to a question is a form of
prompt engineering: it gives the model the domain concepts it needs instead
of relying on it to infer them from a bare id string.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

DEFAULT_FACTS_PATH = Path("data/qKG/facts.tsv")

DOMAIN_CONTEXT = (
    "You are answering questions over a questionnaire knowledge graph. "
    "A Questionnaire contains Questions; each Question may belong to a "
    "Questionnaire. We have 3 Questionnaire SUS, UMARS and Module Evaluation. "
    "Each Questionnaire has a set of Questions, and each Question has a "
    "set of possible Answers. The Questions measure different aspects of "
    "the user experience, such as comprehensibility, recommendation, "
    "satisfaction, usability, or quality). Each Question has possible Answers, and each Answer has a "
    "response code, a label, and an Answer Category. A User completes a "
    "Questionnaire on a Platform, and the answer a user actually gives is "
    "a Response, which links a User, Platform, Questionnaire, Question, "
    "and the selected Answer. A response id such as '217_M010_01' means "
    "the response by user 217 to question M010_01."
    "LLaMA-3.2 is a large language model that can answer questions over this knowledge graph, given the right context."
    "LLaMA-3.2 is a large language model is finetuned with NeSy CoT rules which are mined from this knowledge graph and knows the relationships."
)

Facts = dict[str, dict[str, list[str]]]

_COMPOSITE_RE = re.compile(r"\b\d+_[A-Za-z][A-Za-z0-9]*(?:_\d+)?\b")
_QUESTION_RE = re.compile(r"\b(?:UM\d+|SU\d+_\d+|M\d+_\d+|M\d+)\b")
_ANSWER_CODE_RE = re.compile(r"\b\d{6,}\b")


def load_facts(path: Path = DEFAULT_FACTS_PATH) -> Facts:
    """Load facts.tsv into ``subject -> predicate -> [objects]``."""
    facts: Facts = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if len(row) != 3:
                continue
            subject, predicate, obj = row
            facts.setdefault(subject, {}).setdefault(predicate, []).append(obj)
    return facts


def extract_entities(text: str) -> list[str]:
    """Best-effort extraction of KG entity ids mentioned in free text."""
    found: list[str] = []
    for pattern in (_COMPOSITE_RE, _QUESTION_RE, _ANSWER_CODE_RE):
        found.extend(pattern.findall(text))
    seen: set[str] = set()
    ordered: list[str] = []
    for entity in found:
        if entity not in seen:
            seen.add(entity)
            ordered.append(entity)
    return ordered


def _first(values: dict[str, list[str]], predicate: str) -> str | None:
    matches = values.get(predicate)
    return matches[0] if matches else None


def _describe_question(question_id: str, facts: Facts) -> str:
    info = facts.get(question_id, {})
    parts = [f"Question {question_id}"]

    questionnaire = _first(info, "belongsTo")
    if questionnaire:
        q_info = facts.get(questionnaire, {})
        blurb = _first(q_info, "comment") or _first(q_info, "shortDescription")
        parts.append(f"belongs to questionnaire '{questionnaire}'")
        if blurb:
            parts.append(f"({blurb})")

    category = _first(info, "hasQuestion_Category")
    if category:
        cat_comment = _first(facts.get(category, {}), "comment")
        parts.append(f"and measures the concept '{category}'")
        if cat_comment:
            parts.append(f"- {cat_comment}")

    description = _first(info, "Question_Description")
    if description:
        parts.append(f"Question reference/text: {description}.")

    return " ".join(parts)


def _describe_answer(answer_id: str, facts: Facts) -> str | None:
    info = facts.get(answer_id)
    if not info or "is_Answer_To" not in info:
        return None
    parts = [f"Answer {answer_id}"]
    question_id = _first(info, "is_Answer_To")
    if question_id:
        parts.append(f"is a possible answer to question {question_id}")
    labels = info.get("ResponseLabel")
    if labels:
        parts.append(f"with label(s) {', '.join(labels)}")
    category = _first(info, "hasAnswer_Category")
    if category:
        parts.append(f"categorized as '{category}'")
    code = _first(info, "ResponseCode")
    if code:
        parts.append(f"(response code {code})")
    return " ".join(parts) + "."


def describe_entity(entity: str, facts: Facts) -> str | None:
    """Return a natural-language description of what a KG entity means."""
    entity = entity.strip()
    if not entity:
        return None

    info = facts.get(entity)
    if info and ("hasQuestion_Category" in info or "Question_Description" in info):
        return _describe_question(entity, facts)

    if "_" in entity:
        user_id, _, question_id = entity.partition("_")
        if user_id.isdigit() and question_id in facts:
            return (
                f"{entity} is the response by user {user_id} to question "
                f"{question_id}. {_describe_question(question_id, facts)}"
            )

    if entity.isdigit():
        return _describe_answer(entity, facts)

    return None


def build_concept_context(
    question: str, facts: Facts, entities: list[str] | None = None
) -> str:
    """Build the full concept-enriched context block for a question.

    ``entities`` should be the KG entity ids the question is about (for
    example from a ``q_entity`` column); if not supplied they are guessed
    from the question text with :func:`extract_entities`.
    """
    if entities is None:
        entities = extract_entities(question)

    descriptions = []
    seen: set[str] = set()
    for entity in entities:
        description = describe_entity(entity, facts)
        if description and description not in seen:
            seen.add(description)
            descriptions.append(description)

    parts = [DOMAIN_CONTEXT]
    if descriptions:
        facts_block = "\n".join(f"- {d}" for d in descriptions)
        parts.append("Relevant knowledge graph facts for this question:\n" + facts_block)
    return "\n\n".join(parts)


def build_concept_prompt(
    question: str,
    facts: Facts,
    entities: list[str] | None = None,
    instruction: str = "Answer the question directly and concisely.",
) -> str:
    """Full prompt: domain + concept facts + instruction + question."""
    context = build_concept_context(question, facts, entities)
    return f"{context}\n\n{instruction}\n\nQuestion: {question}"

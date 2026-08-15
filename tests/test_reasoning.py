"""P0 tests for atomic schema loading and fail-closed reasoning."""

import pytest

import reasoning
from contracts import ReasoningStatus
from extraction import relation_key, stable_relation_id
from reasoning import MalOntReasoner


def valid_graph():
    return {
        "status": "success",
        "entities": [
            {"id": "actor-id", "type": "ThreatActor", "name": "APT29"},
            {"id": "malware-id", "type": "Malware", "name": "CozyCar"},
            {"id": "vuln-id", "type": "Vulnerability", "name": "CVE-1"},
        ],
        "relations": [
            {
                "subject": "actor-id",
                "predicate": "Uses",
                "object": "malware-id",
                "provenance": ["chunk-actor", "chunk-shared"],
            },
            {
                "subject": "malware-id",
                "predicate": "Targets",
                "object": "vuln-id",
                "provenance": ["chunk-vuln", "chunk-shared"],
            },
        ],
    }


@pytest.mark.parametrize(
    "mutation, fragment",
    [
        (
            lambda data: data["entities"].append(
                {"id": "actor-id", "type": "ThreatActor", "name": "Duplicate"}
            ),
            "duplicate entity id",
        ),
        (
            lambda data: data["relations"].append(
                {"subject": "missing", "predicate": "Uses", "object": "malware-id"}
            ),
            "dangling subject",
        ),
        (
            lambda data: data["relations"].append(
                {"subject": "actor-id", "predicate": "Unknown", "object": "malware-id"}
            ),
            "predicate is not permitted",
        ),
    ],
)
def test_schema_failure_prevents_all_ontology_mutation(mutation, fragment):
    data = valid_graph()
    mutation(data)
    reasoner = MalOntReasoner()

    result = reasoner.load_extracted_data(data)

    assert result.status is ReasoningStatus.VALIDATION_ERROR
    assert reasoner.entity_map == {}
    assert any(fragment in item for item in result["diagnostics"])


def test_owl_individual_iri_uses_external_id_not_display_name():
    reasoner = MalOntReasoner()
    assert reasoner.load_extracted_data(valid_graph()).status is ReasoningStatus.VALIDATED

    actor = reasoner.entity_map["actor-id"]
    assert "actor_id" in actor.name
    assert "APT29" not in actor.name
    assert actor.label == ["APT29"]


def test_missing_java_is_fail_closed(monkeypatch):
    reasoner_instance = MalOntReasoner()
    reasoner_instance.load_extracted_data(valid_graph())

    def missing_java(*_args, **_kwargs):
        raise FileNotFoundError("java executable not found")

    monkeypatch.setattr(reasoning, "sync_reasoner_pellet", missing_java)
    result = reasoner_instance.verify_and_infer()

    assert result.status is ReasoningStatus.REASONER_UNAVAILABLE
    assert not result


def test_reasoner_reports_new_inference(monkeypatch):
    reasoner_instance = MalOntReasoner()
    reasoner_instance.load_extracted_data(valid_graph())

    def infer_actor_target(*_args, **_kwargs):
        actor = reasoner_instance.entity_map["actor-id"]
        vulnerability = reasoner_instance.entity_map["vuln-id"]
        actor.Targets.append(vulnerability)

    monkeypatch.setattr(reasoning, "sync_reasoner_pellet", infer_actor_target)
    result = reasoner_instance.verify_and_infer()

    assert result.status is ReasoningStatus.CONSISTENT
    assert result["inferred_relations"] == [
        {
            "subject": "actor-id",
            "predicate": "Targets",
            "object": "vuln-id",
            "rule_id": reasoning.ACTOR_TARGET_RULE_ID,
            "proofs": [
                {
                    "support_relation_ids": [
                        stable_relation_id("actor-id", "Uses", "malware-id"),
                        stable_relation_id("malware-id", "Targets", "vuln-id"),
                    ],
                    "support_relation_keys": [
                        relation_key("actor-id", "Uses", "malware-id"),
                        relation_key("malware-id", "Targets", "vuln-id"),
                    ],
                    "provenance": ["chunk-actor", "chunk-shared", "chunk-vuln"],
                }
            ],
            "provenance": ["chunk-actor", "chunk-shared", "chunk-vuln"],
        }
    ]


def test_load_retains_normalized_asserted_relation_provenance():
    reasoner_instance = MalOntReasoner()
    data = valid_graph()
    data["entities"][0]["provenance"] = ["chunk-actor"]

    result = reasoner_instance.load_extracted_data(data)

    assert result.status is ReasoningStatus.VALIDATED
    assert reasoner_instance.entity_records["actor-id"]["provenance"] == [
        "chunk-actor"
    ]
    asserted = reasoner_instance.asserted_relations[
        ("actor-id", "Uses", "malware-id")
    ]
    assert asserted["id"] == stable_relation_id(
        "actor-id", "Uses", "malware-id"
    )
    assert asserted["key"] == relation_key("actor-id", "Uses", "malware-id")
    assert asserted["provenance"] == ["chunk-actor", "chunk-shared"]


def test_all_valid_actor_target_proofs_are_retained(monkeypatch):
    data = valid_graph()
    data["entities"].append(
        {"id": "malware-2", "type": "Malware", "name": "Second"}
    )
    data["relations"].extend(
        [
            {
                "subject": "actor-id",
                "predicate": "Uses",
                "object": "malware-2",
                "provenance": ["chunk-c"],
            },
            {
                "subject": "malware-2",
                "predicate": "Targets",
                "object": "vuln-id",
                "provenance": ["chunk-d"],
            },
        ]
    )
    reasoner_instance = MalOntReasoner()
    reasoner_instance.load_extracted_data(data)

    def infer_actor_target(*_args, **_kwargs):
        reasoner_instance.entity_map["actor-id"].Targets.append(
            reasoner_instance.entity_map["vuln-id"]
        )

    monkeypatch.setattr(reasoning, "sync_reasoner_pellet", infer_actor_target)
    inferred = reasoner_instance.verify_and_infer()["inferred_relations"][0]

    assert len(inferred["proofs"]) == 2
    assert [proof["support_relation_keys"] for proof in inferred["proofs"]] == [
        [
            relation_key("actor-id", "Uses", "malware-2"),
            relation_key("malware-2", "Targets", "vuln-id"),
        ],
        [
            relation_key("actor-id", "Uses", "malware-id"),
            relation_key("malware-id", "Targets", "vuln-id"),
        ],
    ]
    assert inferred["provenance"] == [
        "chunk-actor",
        "chunk-c",
        "chunk-d",
        "chunk-shared",
        "chunk-vuln",
    ]


def test_unruled_inferred_edge_never_receives_invented_support(monkeypatch):
    data = valid_graph()
    data["entities"].append(
        {"id": "campaign-id", "type": "Campaign", "name": "Campaign"}
    )
    reasoner_instance = MalOntReasoner()
    reasoner_instance.load_extracted_data(data)

    def infer_unruled_edge(*_args, **_kwargs):
        reasoner_instance.entity_map["malware-id"].AssociatedWith.append(
            reasoner_instance.entity_map["campaign-id"]
        )

    monkeypatch.setattr(reasoning, "sync_reasoner_pellet", infer_unruled_edge)
    result = reasoner_instance.verify_and_infer()
    inferred = result["inferred_relations"][0]

    assert inferred == {
        "subject": "malware-id",
        "predicate": "AssociatedWith",
        "object": "campaign-id",
    }
    assert any("no asserted support ledger" in item for item in result["diagnostics"])


def test_reasoner_distinguishes_consistency_from_inference(monkeypatch):
    reasoner_instance = MalOntReasoner()
    reasoner_instance.load_extracted_data(valid_graph())
    monkeypatch.setattr(reasoning, "sync_reasoner_pellet", lambda *_args, **_kwargs: None)

    result = reasoner_instance.verify_and_infer()

    assert result.status is ReasoningStatus.CONSISTENT
    assert result["inferred_relations"] == []

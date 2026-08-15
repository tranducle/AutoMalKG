"""P0 tests for extraction status, validation, and canonicalization."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from contracts import ExtractionStatus, StructuredChunk
from extraction import MalOntExtractor, validate_extraction_schema


class FakeClient:
    """Return deterministic responses keyed by exact chunk text."""

    def __init__(self, responses):
        self.responses = responses

    async def generate_response(self, _system_prompt, text):
        response = self.responses[text]
        if isinstance(response, Exception):
            raise response
        return response


def run(coroutine):
    return asyncio.run(coroutine)


def test_malformed_empty_and_client_error_are_distinct():
    extractor = MalOntExtractor(
        FakeClient({"bad": "not-json", "empty": "", "boom": RuntimeError("down")})
    )

    assert run(extractor.extract_from_chunk("bad")).status is ExtractionStatus.MALFORMED
    assert run(extractor.extract_from_chunk("empty")).status is ExtractionStatus.EMPTY
    assert run(extractor.extract_from_chunk("boom")).status is ExtractionStatus.CLIENT_ERROR


def test_typed_client_response_text_is_consumed_without_client_coupling():
    payload = {
        "entities": [
            {
                "id": "m",
                "type": "Malware",
                "name": "Alpha",
                "provenance": ["upstream-record"],
            }
        ],
        "relations": [],
    }
    extractor = MalOntExtractor(
        FakeClient({"typed": SimpleNamespace(text=json.dumps(payload))})
    )

    result = run(extractor.extract_from_chunk("typed"))
    assert result.status is ExtractionStatus.SUCCESS
    assert result["entities"][0]["provenance"] == sorted(
        ["upstream-record", result["chunk_id"]]
    )
    assert result["chunk_registry"][result["chunk_id"]]["source_id"] == "inline"


@pytest.mark.parametrize(
    "payload, diagnostic_fragment",
    [
        (
            {
                "entities": [
                    {"id": "x", "type": "Malware", "name": "Alpha"},
                    {"id": "x", "type": "Malware", "name": "Beta"},
                ],
                "relations": [],
            },
            "duplicate entity id",
        ),
        (
            {
                "entities": [{"id": "x", "type": "Malware", "name": "Alpha"}],
                "relations": [
                    {"subject": "x", "predicate": "Uses", "object": "missing"}
                ],
            },
            "dangling object",
        ),
        (
            {
                "entities": [{"id": "x", "type": "Malware", "name": "Alpha"}],
                "relations": [
                    {"subject": "x", "predicate": "Invented", "object": "x"}
                ],
            },
            "predicate is not permitted",
        ),
    ],
)
def test_schema_rejects_duplicates_dangling_edges_and_unknown_predicates(
    payload, diagnostic_fragment
):
    extractor = MalOntExtractor(FakeClient({"case": json.dumps(payload)}))
    result = run(extractor.extract_from_chunk("case"))

    assert result.status is ExtractionStatus.SCHEMA_INVALID
    assert any(diagnostic_fragment in item for item in result["diagnostics"])
    assert result["entities"] == []
    assert result["relations"] == []


@pytest.mark.parametrize(
    "payload, expected_diagnostic",
    [
        (
            {"entities": [], "relations": [], "unexpected_root": True},
            "root contains unknown key: 'unexpected_root'",
        ),
        (
            {
                "entities": [
                    {
                        "id": "m",
                        "type": "Malware",
                        "name": "Alpha",
                        "unexpected_entity": True,
                    }
                ],
                "relations": [],
            },
            "entities[0] contains unknown key: 'unexpected_entity'",
        ),
        (
            {
                "entities": [
                    {"id": "a", "type": "ThreatActor", "name": "APT29"},
                    {"id": "m", "type": "Malware", "name": "Alpha"},
                ],
                "relations": [
                    {
                        "subject": "a",
                        "predicate": "Uses",
                        "object": "m",
                        "unexpected_relation": True,
                    }
                ],
            },
            "relations[0] contains unknown key: 'unexpected_relation'",
        ),
    ],
)
def test_schema_rejects_unknown_keys_with_typed_path_diagnostics(
    payload, expected_diagnostic
):
    result = run(
        MalOntExtractor(FakeClient({"case": json.dumps(payload)})).extract_from_chunk(
            "case"
        )
    )

    assert result.status is ExtractionStatus.SCHEMA_INVALID
    assert expected_diagnostic in result["diagnostics"]
    assert result["entities"] == []
    assert result["relations"] == []


def test_schema_accepts_every_permitted_root_entity_and_relation_field():
    payload = {
        "entities": [
            {
                "id": "a",
                "type": "ThreatActor",
                "name": "APT29",
                "aliases": ["The Dukes"],
                "provenance": ["source:actor"],
                "evidence": "APT29",
            },
            {
                "id": "m",
                "type": "Malware",
                "name": "CozyCar",
                "aliases": [],
                "provenance": ["source:malware"],
                "evidence": "CozyCar",
            },
        ],
        "relations": [
            {
                "subject": "a",
                "predicate": "Uses",
                "object": "m",
                "provenance": ["source:relation"],
                "evidence": "APT29 uses CozyCar",
            }
        ],
    }

    entities, relations, diagnostics = validate_extraction_schema(payload)

    assert diagnostics == []
    assert len(entities) == 2
    assert len(relations) == 1


def test_raw_schema_does_not_treat_internal_result_envelope_fields_as_llm_fields():
    _, _, diagnostics = validate_extraction_schema(
        {"status": "success", "entities": [], "relations": []}
    )

    assert diagnostics == ["root contains unknown key: 'status'"]


def test_raw_relation_rejects_internal_canonical_id_and_key_fields():
    payload = {
        "entities": [
            {"id": "a", "type": "ThreatActor", "name": "APT29"},
            {"id": "m", "type": "Malware", "name": "CozyCar"},
        ],
        "relations": [
            {
                "id": "rel_not_allowed_from_llm",
                "key": "a|Uses|m",
                "subject": "a",
                "predicate": "Uses",
                "object": "m",
            }
        ],
    }

    _, _, diagnostics = validate_extraction_schema(payload)

    assert "relations[0] contains unknown key: 'id'" in diagnostics
    assert "relations[0] contains unknown key: 'key'" in diagnostics


def test_internal_result_envelope_accepts_verified_canonical_relation_fields():
    raw_payload = {
        "entities": [
            {"id": "a", "type": "ThreatActor", "name": "APT29"},
            {"id": "m", "type": "Malware", "name": "CozyCar"},
        ],
        "relations": [
            {"subject": "a", "predicate": "Uses", "object": "m"}
        ],
    }
    entities, relations, diagnostics = validate_extraction_schema(raw_payload)
    assert diagnostics == []

    validated_entities, validated_relations, envelope_diagnostics = (
        validate_extraction_schema(
            {
                "status": "success",
                "diagnostics": [],
                "entities": entities,
                "relations": relations,
            },
            allow_result_envelope=True,
        )
    )

    assert envelope_diagnostics == []
    assert validated_entities == entities
    assert validated_relations == relations


def test_canonicalization_deduplicates_and_unions_relation_provenance():
    first = StructuredChunk.create(
        text="first", source_id="report.pdf", index=0, start_word=0, end_word=1
    )
    second = StructuredChunk.create(
        text="second", source_id="report.pdf", index=1, start_word=1, end_word=2
    )
    responses = {
        "first": json.dumps(
            {
                "entities": [
                    {"id": "a", "type": "ThreatActor", "name": "APT 29"},
                    {"id": "m", "type": "Malware", "name": "CozyCar"},
                ],
                "relations": [{"subject": "a", "predicate": "Uses", "object": "m"}],
            }
        ),
        "second": json.dumps(
            {
                "entities": [
                    {"id": "actor", "type": "ThreatActor", "name": "APT   29"},
                    {"id": "mal", "type": "Malware", "name": "cozycar"},
                ],
                "relations": [
                    {"subject": "actor", "predicate": "Uses", "object": "mal"}
                ],
            }
        ),
    }
    extractor = MalOntExtractor(FakeClient(responses))

    result = run(extractor.process_document_chunks([first, second]))

    assert result.status is ExtractionStatus.SUCCESS
    assert len(result["entities"]) == 2
    assert len(result["relations"]) == 1
    assert result["relations"][0]["provenance"] == sorted(
        [first.chunk_id, second.chunk_id]
    )
    assert result["relations"][0]["id"].startswith("rel_")
    assert result["relations"][0]["key"].count("|") == 2
    assert result["chunk_registry"] == {
        first.chunk_id: {
            key: value for key, value in first.to_dict().items() if key != "text"
        },
        second.chunk_id: {
            key: value for key, value in second.to_dict().items() if key != "text"
        },
    }
    assert all(
        entity["provenance"] == sorted([first.chunk_id, second.chunk_id])
        for entity in result["entities"]
    )

    reversed_result = run(
        MalOntExtractor(FakeClient(responses)).process_document_chunks([second, first])
    )
    assert reversed_result == result


def test_same_canonical_name_with_different_types_is_diagnosed_not_merged():
    responses = {
        "one": json.dumps(
            {
                "entities": [{"id": "x", "type": "Malware", "name": "Orion"}],
                "relations": [],
            }
        ),
        "two": json.dumps(
            {
                "entities": [{"id": "y", "type": "Campaign", "name": "ORION"}],
                "relations": [],
            }
        ),
    }
    result = run(MalOntExtractor(FakeClient(responses)).process_document_chunks(["one", "two"]))

    assert result.status is ExtractionStatus.SUCCESS
    assert len(result["entities"]) == 2
    assert any("canonical-name type collision" in item for item in result["diagnostics"])


def test_partial_chunk_failure_is_not_promoted_to_success():
    responses = {
        "good": json.dumps(
            {
                "entities": [{"id": "x", "type": "Malware", "name": "Alpha"}],
                "relations": [],
            }
        ),
        "bad": "not-json",
    }
    result = run(MalOntExtractor(FakeClient(responses)).process_document_chunks(["good", "bad"]))

    assert result.status is ExtractionStatus.PARTIAL
    assert len(result["entities"]) == 1

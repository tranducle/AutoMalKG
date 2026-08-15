"""Tests for the offline nearest-alternative and reachability experiment."""

from __future__ import annotations

import asyncio
import json

from contracts import ExtractionStatus, StructuredChunk
from extraction import MalOntExtractor
import run_stage_reachability_ablation as ablation


def test_parse_only_uses_same_code_fence_normalization_as_production():
    accepted, value, diagnostics = ablation.parse_only(
        '```json\n{"entities": [], "relations": []}\n```'
    )
    assert accepted is True
    assert value == {"entities": [], "relations": []}
    assert diagnostics == []


def test_strict_json_schema_is_structural_not_referential_or_type_aware():
    wrong_signature = json.dumps(
        {
            "entities": [
                {"id": "a", "type": "ThreatActor", "name": "A"},
                {"id": "v", "type": "Vulnerability", "name": "V"},
            ],
            "relations": [
                {"subject": "a", "predicate": "Targets", "object": "v"}
            ],
        }
    )
    accepted, _, diagnostics = ablation.strict_schema_validate(wrong_signature)
    assert accepted is True
    assert diagnostics == []

    chunk = StructuredChunk.create(
        text="fixture-prompt", source_id="test", index=0, start_word=0, end_word=1
    )
    client = ablation.MappingReplayClient({"fixture-prompt": wrong_signature})
    result = asyncio.run(MalOntExtractor(client).process_document_chunks([chunk]))
    assert result.status is ExtractionStatus.SCHEMA_INVALID
    assert any("subject type ThreatActor is invalid for Targets" in item for item in result["diagnostics"])


def test_fixture_ledger_is_deterministic_and_has_independent_violation_labels():
    first = ablation.build_structural_fixtures()
    second = ablation.build_structural_fixtures()
    assert first == second
    assert len(first) == ablation.EXPECTED_SYNTHETIC_FIXTURES == 29
    assert len({row["case_id"] for row in first}) == len(first)
    assert (
        sum(row["known_contract_violation"] for row in first)
        == ablation.EXPECTED_KNOWN_VIOLATIONS
        == 26
    )
    assert sum(row["expected_derivations"] for row in first) == 1
    assert next(row for row in first if row["case_id"] == "fixture-client-error")["responses"] == []


def test_unknown_key_fixtures_are_independent_prespecified_violations():
    fixtures = {
        row["case_id"]: row for row in ablation.build_structural_fixtures()
    }

    for case_id in (
        "fixture-unknown-root-key",
        "fixture-unknown-entity-key",
        "fixture-unknown-relation-key",
    ):
        row = fixtures[case_id]
        assert row["family"] == "unknown_field"
        assert row["known_contract_violation"] is True
        assert row["expected_failure_stage"] == "json_schema"
        accepted, _, diagnostics = ablation.strict_schema_validate(row["responses"][0])
        assert accepted is False
        assert any("Additional properties are not allowed" in item for item in diagnostics)


def test_production_provenance_requires_registry_resolution():
    extraction = {
        "entities": [
            {"id": "a", "provenance": ["chunk-a"]},
            {"id": "b", "provenance": ["missing"]},
        ],
        "relations": [],
        "chunk_registry": {"chunk-a": {"source_id": "test"}},
    }
    complete, total = ablation._production_provenance(extraction)
    assert (complete, total) == (1, 2)


def test_summary_counts_silent_acceptance_only_on_prespecified_violations():
    def row(case_id: str, violation: bool, accepted: bool) -> dict:
        methods = {}
        for method in ablation.METHODS:
            methods[method] = {
                "accepted": accepted,
                "provenance_complete": 0,
                "provenance_total": 0,
                "derivations_with_complete_proof": 0,
            }
        return {
            "case_id": case_id,
            "source_kind": "synthetic_fixture",
            "known_contract_violation": violation,
            "expected_derivations": 0,
            "observed_failure_stage": None,
            "methods": methods,
        }

    summary = ablation.summarize_observations(
        [row("bad", True, True), row("good", False, True)]
    )
    metrics = summary["synthetic_fixture"]["methods"]["json_parse_only"]
    assert metrics["accepted"] == 2
    assert metrics["known_violation_cases"] == 1
    assert metrics["silent_acceptance_of_known_violations"] == 1


def test_reachability_summary_distinguishes_pipeline_from_integration():
    rows = [
        {
            "passed": True,
            "expected_status": "inconsistent",
            "axiom_or_failure_family": "AllDisjoint",
            "production_input_reachable": False,
            "reasoner_layer_injection": True,
            "member": "A vs B",
        },
        {
            "passed": True,
            "expected_status": "consistent",
            "axiom_or_failure_family": "ObjectPropertyDomainRange",
            "production_input_reachable": False,
            "reasoner_layer_injection": True,
            "member": "Targets.domain",
        },
    ]
    summary = ablation.summarize_reachability(rows)
    assert summary["owl_inconsistency_classes"] == 1
    assert summary["owl_inconsistency_classes_reachable_from_production_input"] == 0
    assert summary["owl_inconsistency_classes_reachable_by_reasoner_integration_injection"] == 1
    assert summary["non_discriminating_axioms"] == ["Targets.domain"]


def test_fault_labels_separate_unavailable_from_generic_reasoner_error():
    assert ablation.MalOntReasoner._reasoner_unavailable(
        FileNotFoundError("synthetic missing Java")
    ) is True
    assert ablation.MalOntReasoner._reasoner_unavailable(
        RuntimeError("synthetic execution fault")
    ) is False


def test_input_parity_allows_prespecified_client_error_without_response_bytes():
    observations = []
    for case_id, hashes, stage in (
        ("hosted", ["a" * 64], None),
        ("client-error", [], "client_error"),
    ):
        observations.append(
            {
                "case_id": case_id,
                "source_kind": (
                    "frozen_hosted_response" if case_id == "hosted" else "synthetic_fixture"
                ),
                "response_sha256": hashes,
                "expected_failure_stage": stage,
                "methods": {method: {} for method in ablation.METHODS},
            }
        )
    metrics = {
        "summaries": {
            "synthetic_fixture": {
                "methods": {
                    method: {
                        "accepted": 1,
                        "rejected": 1,
                        "silent_acceptance_of_known_violations": 0,
                    }
                    for method in ablation.METHODS
                }
            }
        }
    }
    g1, g2, fairness = ablation.build_baseline_prerequisite_gates(
        observations, metrics
    )
    assert g1["verdict"] == "PASS"
    assert g2["verdict"] == "PASS"
    assert fairness["verdict"] == "PASS_WITH_WARNINGS"


def test_fixture_only_result_gate_requires_zero_hosted_rows():
    observations = [
        {
            "case_id": "fixture-valid",
            "source_kind": "synthetic_fixture",
            "known_contract_violation": False,
            "response_sha256": ["a" * 64],
            "hosted_call_attempts": 0,
            "methods": {
                method: {"accepted": True} for method in ablation.METHODS
            },
        }
    ]
    metrics = {"hosted_call_attempts": 0}

    _, fixture_gate = ablation.build_gates(
        observations, [], metrics, fixtures_only=True
    )
    _, combined_gate = ablation.build_gates(
        observations, [], metrics, fixtures_only=False
    )

    assert fixture_gate["verdict"] == "PASS"
    assert fixture_gate["pass_criteria"]["frozen_hosted_cases"] == 0
    assert fixture_gate["metrics"]["execution_scope"] == "fixtures_only"
    assert combined_gate["verdict"] == "FAIL_REPAIR"
    assert "hosted_case_count_not_126_for_execution_scope" in combined_gate["failed_criteria"]


def test_fixture_only_cli_flag_is_explicit():
    args = ablation.parse_args(["--fixtures-only", "--output-dir", "run"])
    assert args.fixtures_only is True
    assert str(args.output_dir) == "run"

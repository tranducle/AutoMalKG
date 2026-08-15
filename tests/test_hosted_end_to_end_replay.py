"""Tests for exact, fail-closed replay of stored hosted responses."""

from __future__ import annotations

import asyncio
import json

import pytest

from contracts import ExtractionResult, ExtractionStatus
from experiment_support import sha256_file, write_json
import run_hosted_end_to_end_replay as replay


def test_replay_client_returns_frozen_text_once_and_never_falls_back():
    client = replay.ReplayClient('{"entities": [], "relations": []}')
    first = asyncio.run(client.generate_response("ignored", "ignored"))
    assert first == '{"entities": [], "relations": []}'
    assert client.call_count == 1
    with pytest.raises(replay.ReplayContractError, match="exactly once"):
        asyncio.run(client.generate_response("ignored", "ignored"))


def test_stage_classification_preserves_valid_empty_as_schema_pass():
    empty = ExtractionResult(
        ExtractionStatus.EMPTY,
        diagnostics=["valid response contains no entities or relations"],
    )
    stages = replay.classify_extraction_stages(empty)
    assert stages["json_parse"]["status"] == "passed"
    assert stages["schema_validation"]["status"] == "passed"
    assert stages["canonicalization"]["status"] == "completed_empty"
    assert stages["nonempty_extraction_gate"]["status"] == "failed"

    invalid = ExtractionResult(
        ExtractionStatus.SCHEMA_INVALID,
        diagnostics=["entities[0].type is not permitted"],
    )
    stages = replay.classify_extraction_stages(invalid)
    assert stages["json_parse"]["status"] == "passed"
    assert stages["schema_validation"]["status"] == "failed"
    assert stages["canonicalization"]["status"] == "blocked"


def test_attrition_counts_empty_separately_from_nonempty_graph():
    def row(extraction_status: str, canonical: str, serialized: bool) -> dict:
        passed = "passed" if serialized else "blocked"
        return {
            "condition": "typed_evidence",
            "stages": {
                "source_integrity": {"status": "passed"},
                "json_parse": {"status": "passed"},
                "schema_validation": {"status": "passed"},
                "canonicalization": {"status": canonical},
                "nonempty_extraction_gate": {
                    "status": "passed" if serialized else "failed",
                    "extraction_status": extraction_status,
                },
                "ontology_load": {"status": passed},
                "pellet": {
                    "status": passed,
                    "reasoning_status": "consistent" if serialized else "not_run",
                },
                "inference_ledger": {
                    "status": passed,
                    "inferred_relation_count": 0,
                    "relations_with_proofs": 0,
                },
                "serialization": {"status": passed},
            },
        }

    metrics = replay.compute_attrition(
        [row("empty", "completed_empty", False), row("success", "passed", True)]
    )
    assert metrics["stage_counts"]["canonicalization_completed"] == 2
    assert metrics["stage_counts"]["nonempty_canonical_graph"] == 1
    assert metrics["stage_counts"]["serialized_verified_artifacts"] == 1


def test_manifest_verifier_detects_output_mutation(tmp_path):
    payload = tmp_path / "result.json"
    write_json(payload, {"value": 1})
    manifest = {
        "inputs": [],
        "outputs": [
            {
                "path": "result.json",
                "bytes": payload.stat().st_size,
                "sha256": sha256_file(payload),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert replay.verify_manifest_hashes(manifest_path, base=tmp_path)["status"] == "PASS"

    write_json(payload, {"value": 2})
    audit = replay.verify_manifest_hashes(manifest_path, base=tmp_path)
    assert audit["status"] == "FAIL"
    assert audit["failures"] == [
        {"path": "result.json", "reason": "sha256_mismatch"}
    ]

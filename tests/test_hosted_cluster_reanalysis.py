"""Tests for the offline context-clustered hosted-run reanalysis."""

from __future__ import annotations

import pytest

from experiment_support import sha256_file, write_json
import reanalyze_hosted_clusters as clustered


def _prediction(entity_name: str = "APT29", evidence: str = "APT29") -> dict:
    return {
        "entities": [
            {
                "id": "arbitrary-id",
                "type": "ThreatActor",
                "name": entity_name,
                "evidence": evidence,
            }
        ],
        "relations": [],
    }


def test_canonical_prediction_ignores_arbitrary_ids_and_hashes_evidence():
    first = clustered.canonical_prediction(_prediction())
    second_payload = _prediction()
    second_payload["entities"][0]["id"] = "different-id"
    second = clustered.canonical_prediction(second_payload)
    assert first["comparable"] is True
    assert first["entity_set"] == second["entity_set"]
    assert first["evidence_records"] == second["evidence_records"]
    assert "APT29" not in next(iter(first["evidence_records"]))


def test_set_summary_defines_empty_empty_jaccard_as_one():
    summary = clustered._set_summary(set(), set())
    assert summary["jaccard"] == 1.0
    assert summary["additions"] == []
    assert summary["deletions"] == []


def test_paired_cluster_bootstrap_resamples_context_differences_deterministically():
    rows = [
        {"conditions": {"a": {"valid_fraction": 1.0}, "b": {"valid_fraction": 0.0}}},
        {"conditions": {"a": {"valid_fraction": 0.0}, "b": {"valid_fraction": 0.0}}},
    ]
    first = clustered.paired_cluster_bootstrap(rows, "a", "b", seed=7, replicates=1000)
    second = clustered.paired_cluster_bootstrap(rows, "a", "b", seed=7, replicates=1000)
    assert first == second
    assert first["cluster_count"] == 2
    assert first["point_difference"] == pytest.approx(0.5)
    assert first["percentile_95_interval"] == [0.0, 1.0]


def test_injection_pairs_align_sample_and_repeat_and_detect_displacement():
    cases = []
    predictions = []
    raw = []
    for condition, prediction, status in (
        ("typed_evidence", _prediction(), "valid"),
        ("typed_evidence_injection", {"entities": [], "relations": []}, "valid"),
    ):
        case_id = f"sample__{condition}__r1"
        cases.append(
            {
                "case_id": case_id,
                "sample_id": "sample",
                "condition": condition,
                "repeat": 1,
                "attack_canary": "CANARY" if condition.endswith("injection") else None,
            }
        )
        predictions.append(
            {
                "case_id": case_id,
                "sample_id": "sample",
                "condition": condition,
                "repeat": 1,
                "group": "g",
                "context_sha256": "c",
                "status": status,
                "prediction": prediction,
                "prediction_sha256": condition,
                "attack_success": False,
            }
        )
        raw.append({"case_id": case_id, "text": "{}"})
    pair = clustered.build_injection_pairs(cases, predictions, raw)[0]
    assert pair["clean_nonempty_to_injected_empty"] is True
    assert pair["semantic_output_changed"] is True
    assert pair["entity_sets"]["deletions"] == ["entity|threatactor|apt29"]
    assert pair["canary"]["present_in_injected_raw_text"] is False


def test_semantic_template_inventory_distinguishes_family_scope_from_labels():
    inventory = clustered.semantic_template_inventory(
        [
            {"family": "swrl_control", "variant": "licensed_chain", "graph_size": 1},
            {"family": "scale", "variant": "licensed_chain", "graph_size": 100},
            {"family": "scale", "variant": "licensed_chain", "graph_size": 300},
        ]
    )
    assert inventory["family_scoped_template_count"] == 2
    assert inventory["unique_variant_label_count"] == 1
    assert inventory["scale_graph_sizes"] == [100, 300]


def test_output_manifest_verifier_detects_mutation(tmp_path):
    payload = tmp_path / "metrics.json"
    write_json(payload, {"value": 1})
    manifest = {
        "inputs": [],
        "outputs": [
            {"path": "metrics.json", "bytes": payload.stat().st_size, "sha256": sha256_file(payload)}
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    assert clustered.verify_output_manifest(manifest_path, base=tmp_path)["status"] == "PASS"
    write_json(payload, {"value": 2})
    audit = clustered.verify_output_manifest(manifest_path, base=tmp_path)
    assert audit["status"] == "FAIL"
    assert audit["failures"] == [{"path": "metrics.json", "reason": "sha256_mismatch"}]

"""Unit tests for the bounded hosted-model stress harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_hosted_stress as stress


def _row(context: str) -> dict:
    return {"context": context, "technique": None, "metadata": {}}


def _write_rows(path: Path, *contexts: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([_row(context) for context in contexts]), encoding="utf-8")


def test_sampling_is_deterministic_stratified_and_collapses_apt29_annotators(tmp_path):
    root = tmp_path / "CTI-HAL"
    for group in ("fin7", "sandworm"):
        _write_rows(root / "data" / group / "ONE.json", f"{group} alpha", f"{group} beta")
        _write_rows(root / "data" / group / "TWO.json", f"{group} gamma")
    _write_rows(root / "data" / "apt29" / "annotator L" / "REPORT.json", "L alpha", "shared")
    _write_rows(root / "data" / "apt29" / "annotator S" / "REPORT.json", "S duplicate", "shared")
    _write_rows(root / "data" / "apt29" / "annotator S" / "OTHER.json", "other alpha", "other beta")

    first = stress.sample_ctihal_contexts(root, seed=17, contexts_per_group=2)
    second = stress.sample_ctihal_contexts(root, seed=17, contexts_per_group=2)

    assert first == second
    assert {row["group"] for row in first} == {"apt29", "fin7", "sandworm"}
    assert all(sum(item["group"] == group for item in first) == 2 for group in {"apt29", "fin7", "sandworm"})
    apt_paths = {row["source_path"] for row in first if row["group"] == "apt29"}
    assert not any("annotator S/REPORT.json" in path for path in apt_paths)


def test_case_expansion_preregisters_conditions_and_repeats():
    sample = {
        "sample_id": "fin7-abc",
        "group": "fin7",
        "source_path": "data/fin7/A.json",
        "source_row_index": 0,
        "source_sha256": "a" * 64,
        "context": "FIN7 used malware.",
        "context_sha256": "b" * 64,
        "rank_sha256": "c" * 64,
    }
    cases = stress.generate_cases([sample], repeats=3)
    assert len(cases) == 9
    assert {row["condition"] for row in cases} == set(stress.CONDITIONS)
    assert {row["repeat"] for row in cases} == {1, 2, 3}
    injected = [row for row in cases if row["condition"] == "typed_evidence_injection"]
    assert all(stress.ATTACK_CANARY in row["user_prompt"] for row in injected)
    assert all(stress.ATTACK_CANARY not in row["context"] for row in injected)


def test_validation_checks_schema_endpoints_signatures_and_exact_evidence():
    base = {"condition": "typed_evidence", "context": "APT29 uses CozyCar."}
    valid = json.dumps(
        {
            "entities": [
                {"id": "a", "type": "ThreatActor", "name": "APT29", "evidence": "APT29"},
                {"id": "m", "type": "Malware", "name": "CozyCar", "evidence": "CozyCar"},
            ],
            "relations": [
                {"subject": "a", "predicate": "Uses", "object": "m", "evidence": "APT29 uses CozyCar"}
            ],
        }
    )
    assert stress.evaluate_response(base, valid)["status"] == "valid"

    bad_evidence = json.loads(valid)
    bad_evidence["entities"][0]["evidence"] = "apt29"
    assert stress.evaluate_response(base, json.dumps(bad_evidence))["status"] == "evidence_invalid"

    bad_signature = json.loads(valid)
    bad_signature["relations"][0]["subject"] = "m"
    assert stress.evaluate_response(base, json.dumps(bad_signature))["status"] == "schema_invalid"

    dangling = json.loads(valid)
    dangling["relations"][0]["object"] = "missing"
    assert stress.evaluate_response(base, json.dumps(dangling))["status"] == "schema_invalid"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"unexpected_root": True}),
        lambda payload: payload["entities"][0].update({"unexpected_entity": True}),
        lambda payload: payload["relations"][0].update(
            {"unexpected_relation": True}
        ),
    ],
)
def test_hosted_validation_uses_the_closed_shared_schema_contract(mutate):
    case = {"condition": "typed_evidence", "context": "APT29 uses CozyCar."}
    payload = {
        "entities": [
            {
                "id": "a",
                "type": "ThreatActor",
                "name": "APT29",
                "evidence": "APT29",
            },
            {
                "id": "m",
                "type": "Malware",
                "name": "CozyCar",
                "evidence": "CozyCar",
            },
        ],
        "relations": [
            {
                "subject": "a",
                "predicate": "Uses",
                "object": "m",
                "evidence": "APT29 uses CozyCar",
            }
        ],
    }
    mutate(payload)

    result = stress.evaluate_response(case, json.dumps(payload))

    assert result["status"] == "schema_invalid"
    assert any("contains unknown key" in item for item in result["diagnostics"])


def test_attack_success_definition_avoids_valid_json_evidence_false_positive():
    case = {
        "condition": "typed_evidence_injection",
        "context": "The report mentions malware.",
    }
    attacked = stress.evaluate_response(case, stress.ATTACK_CANARY)
    assert attacked["attack_success"] is True
    assert attacked["status"] == "malformed_json"

    valid = stress.evaluate_response(case, '{"entities": [], "relations": []}')
    assert valid["attack_success"] is False
    assert valid["status"] == "valid"


def test_projection_aborts_over_budget(tmp_path, monkeypatch):
    root = tmp_path / "CTI-HAL"
    _write_rows(root / "data" / "fin7" / "ONE.json", "one", "two")
    monkeypatch.setattr(
        stress,
        "fetch_pricing_snapshot",
        lambda: {
            "pricing_per_token_usd": {"prompt": "1", "completion": "1"},
            "model_id": stress.MODEL_ID,
        },
    )
    with pytest.raises(stress.ExperimentConfigurationError, match="exceeds budget"):
        stress.prepare_run(
            dataset_root=root,
            output_root=tmp_path / "runs",
            seed=1,
            contexts_per_group=2,
            repeats=3,
            budget_usd=1.0,
        )


def test_future_hosted_calls_fail_closed_without_completion_cap_support():
    class LegacyClient:
        async def generate_response(self, system_prompt, user_prompt, temperature=0.0):
            raise AssertionError("uncapped provider call must not be reached")

    with pytest.raises(stress.ExperimentConfigurationError, match="cannot enforce"):
        import asyncio

        asyncio.run(
            stress._generate_with_completion_cap(
                LegacyClient(),
                system_prompt="system",
                user_prompt="user",
                temperature=0.0,
                max_completion_tokens=32,
            )
        )


def test_manifest_inputs_resolve_and_hash_against_dataset_root(tmp_path):
    dataset_root = tmp_path / "CTI-HAL"
    source = dataset_root / "data" / "apt29" / "annotator L" / "SEA.json"
    _write_rows(source, "APT29 context")
    cases = [{"source_path": "data/apt29/annotator L/SEA.json"}]

    resolved = stress.resolve_dataset_inputs(cases, dataset_root)
    records = stress.build_manifest(
        run_id="test",
        command=["test"],
        seed=1,
        started_at="2026-08-09T00:00:00+00:00",
        finished_at="2026-08-09T00:00:01+00:00",
        inputs=resolved,
        outputs=[],
        base=stress.PAPER_ROOT,
        dependencies=(),
    )["inputs"]

    assert resolved == [source.resolve()]
    assert len(records) == 1
    assert records[0]["sha256"] == stress.sha256_file(source)


def test_metrics_compute_exact_stability_latency_tokens_and_cost():
    cases = []
    predictions = []
    for condition in stress.CONDITIONS:
        for repeat in (1, 2, 3):
            cases.append({"condition": condition, "sample_id": "s", "repeat": repeat})
            predictions.append(
                {
                    "condition": condition,
                    "sample_id": "s",
                    "repeat": repeat,
                    "status": "valid",
                    "raw_text_sha256": "raw",
                    "prediction_sha256": "prediction",
                    "attack_success": False,
                    "latency_seconds": float(repeat),
                    "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            )
    pricing = {"pricing_per_token_usd": {"prompt": "0.001", "completion": "0.002"}}
    metrics = stress.compute_metrics(cases, predictions, pricing)
    assert metrics["token_totals"]["total_tokens"] == 135
    assert metrics["estimated_cost_usd_from_reported_tokens"] == pytest.approx(0.18)
    condition = metrics["conditions"]["minimal_json"]
    assert condition["exact_raw_repeat_stability_rate"] == 1.0
    assert condition["exact_prediction_repeat_stability_rate"] == 1.0
    assert condition["latency_seconds"] == {"median": 2.0, "iqr": 1.0, "p95": 2.9}

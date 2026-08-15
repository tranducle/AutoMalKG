"""Tests for deterministic, observation-level acceptance-loop harnesses."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import build_ctihal_manifest
import run_semantic_benchmark
from experiment_support import build_manifest, write_json


def test_case_generation_is_seeded_and_balanced():
    kwargs = dict(
        seed=17,
        consistency_cases_per_class=3,
        swrl_controls_per_class=2,
        scale_sizes=(1, 4),
        scale_repeats=2,
    )
    first = run_semantic_benchmark.generate_cases(**kwargs)
    second = run_semantic_benchmark.generate_cases(**kwargs)

    assert first == second
    consistency = [row for row in first if row["family"] == "consistency"]
    assert sum(row["expected_consistency"] == "CONSISTENT" for row in consistency) == 3
    assert sum(row["expected_consistency"] == "INCONSISTENT" for row in consistency) == 3
    controls = [row for row in first if row["family"] == "swrl_control"]
    assert sum(row["expected_edge"] is True for row in controls) == 2
    assert sum(row["expected_edge"] is False for row in controls) == 2


def test_metrics_keep_unverified_out_of_accuracy_denominator():
    predictions = [
        {
            "case_id": "c1",
            "family": "consistency",
            "warmup": False,
            "expected_consistency": "CONSISTENT",
            "observed_consistency": "CONSISTENT",
            "graph_size": 1,
        },
        {
            "case_id": "c2",
            "family": "consistency",
            "warmup": False,
            "expected_consistency": "INCONSISTENT",
            "observed_consistency": "INCONSISTENT",
            "graph_size": 1,
        },
        {
            "case_id": "c3",
            "family": "consistency",
            "warmup": False,
            "expected_consistency": "CONSISTENT",
            "observed_consistency": "UNVERIFIED",
            "graph_size": 1,
        },
        {
            "case_id": "p",
            "family": "swrl_control",
            "warmup": False,
            "expected_consistency": "CONSISTENT",
            "observed_consistency": "CONSISTENT",
            "expected_edge": True,
            "observed_edge": True,
            "predicted_edges": [{"supporting_edge_ids": ["e1", "e2"]}],
            "graph_size": 1,
        },
        {
            "case_id": "n",
            "family": "swrl_control",
            "warmup": False,
            "expected_consistency": "CONSISTENT",
            "observed_consistency": "CONSISTENT",
            "expected_edge": False,
            "observed_edge": False,
            "predicted_edges": [],
            "graph_size": 1,
        },
    ]
    metrics = run_semantic_benchmark.compute_metrics(predictions)

    assert metrics["consistency"]["attempted"] == 3
    assert metrics["consistency"]["verified"] == 2
    assert metrics["consistency"]["unverified"] == 1
    assert metrics["consistency"]["status_accuracy"] == 1.0
    assert metrics["swrl_controls"]["edge_f1"] == 1.0
    assert metrics["swrl_controls"]["inferred_edges_with_support_ledger"] == 1


def test_metrics_exclude_warmup_only_scale_from_reported_sizes():
    predictions = [
        {
            "case_id": "warmup",
            "family": "scale",
            "warmup": True,
            "expected_consistency": "CONSISTENT",
            "observed_consistency": "CONSISTENT",
            "expected_edge": True,
            "observed_edge": True,
            "predicted_edges": [],
            "graph_size": 1,
            "runtime_ms": 500.0,
        },
        {
            "case_id": "measured",
            "family": "scale",
            "warmup": False,
            "expected_consistency": "CONSISTENT",
            "observed_consistency": "CONSISTENT",
            "expected_edge": True,
            "observed_edge": True,
            "predicted_edges": [],
            "graph_size": 10,
            "runtime_ms": 600.0,
        },
    ]

    metrics = run_semantic_benchmark.compute_metrics(predictions)

    assert "1" not in metrics["local_scaling"]
    assert metrics["local_scaling"]["10"]["attempted_repetitions"] == 1


def test_ctihal_inventory_audits_annotations_duplicates_and_license(tmp_path):
    root = tmp_path / "CTI-HAL"
    (root / "data" / "apt29" / "annotator L").mkdir(parents=True)
    (root / "data" / "apt29" / "annotator S").mkdir(parents=True)
    (root / "annotations" / "apt29" / "annotator L").mkdir(parents=True)
    (root / "annotations" / "apt29" / "annotator S").mkdir(parents=True)
    row = {
        "context": "PowerShell",
        "technique": "T1059.001",
        "metadata": {"tactic": ["TA0002"]},
    }
    for annotator in ("annotator L", "annotator S"):
        write_json(root / "data" / "apt29" / annotator / "POWER.json", [row])
        (root / "annotations" / "apt29" / annotator / "apt29-power.md").write_text(
            "annotation", encoding="utf-8"
        )

    inventory, observations, failures = build_ctihal_manifest.inventory_ctihal(
        root, expected_commit=None
    )

    assert not failures
    assert inventory["counts"]["json_files"] == 2
    assert inventory["counts"]["json_rows"] == 2
    assert inventory["annotator_audit"]["apt29_independently_double_annotated_pairs"] == 1
    assert inventory["duplicate_audit"]["exact_duplicate_row_excess"] == 1
    assert inventory["license_audit"]["repository_license_detected"] is False
    assert len(observations) == 2


def test_manifest_hashes_material_outputs(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    write_json(input_path, {"input": 1})
    write_json(output_path, {"output": 2})
    manifest = build_manifest(
        run_id="test",
        command=["python", "test.py"],
        seed=7,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        inputs=[input_path],
        outputs=[output_path],
        base=tmp_path,
    )

    assert manifest["inputs"][0]["path"] == "input.json"
    assert len(manifest["inputs"][0]["sha256"]) == 64
    assert manifest["outputs"][0]["path"] == "output.json"
    assert manifest["hash_policy"]["manifest_self_hash"].startswith("excluded")


def test_scripts_are_callable_outside_project_working_directory(tmp_path):
    script = Path(run_semantic_benchmark.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--scale-sizes" in completed.stdout

#!/usr/bin/env python3
"""Replay saved hosted responses through the frozen production pipeline.

No network client is constructed by this module.  Each saved response is
injected into :class:`MalOntExtractor` through :class:`ReplayClient`, then the
ordinary fail-closed ``run_pipeline`` orchestration performs extraction,
canonicalization, ontology loading, Pellet reasoning, proof-ledger generation,
and conditional serialization in an isolated ontology world.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import owlready2 as owl

from contracts import ExtractionStatus, ReasoningStatus, StructuredChunk
from experiment_support import (
    DEFAULT_RUNS_ROOT,
    PAPER_ROOT,
    build_manifest,
    resolve_java,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from extraction import MalOntExtractor
from main_pipeline import run_pipeline
from reasoning import MalOntReasoner


DEFAULT_SOURCE_RUN = (
    DEFAULT_RUNS_ROOT / "hosted_stress_20260809T211342Z"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_RUNS_ROOT / "hosted_end_to_end_replay_20260809"
)
STANDARD_GATE_SCRIPT = (
    PAPER_ROOT.parents[1]
    / ".agents"
    / "skills"
    / "scientific-experiment-gates"
    / "scripts"
    / "gate_result_integrity.py"
)
REQUIRED_SOURCE_FILES = (
    "cases.jsonl",
    "raw_responses.jsonl",
    "predictions.jsonl",
    "manifest.json",
)


class ReplayContractError(RuntimeError):
    """Raised when saved evidence cannot support an exact offline replay."""


class ReplayClient:
    """Minimal asynchronous client that returns one frozen response verbatim."""

    def __init__(self, response_text: str) -> None:
        if not isinstance(response_text, str):
            raise TypeError("response_text must be a string")
        self.response_text = response_text
        self.call_count = 0

    async def generate_response(self, _system_prompt: str, _user_prompt: str) -> str:
        """Return the frozen response; a second request fails closed."""

        self.call_count += 1
        if self.call_count != 1:
            raise ReplayContractError("replay client may be consumed exactly once")
        return self.response_text


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ReplayContractError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _index_unique(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ReplayContractError(f"{label} contains a missing/invalid case_id")
        if case_id in indexed:
            raise ReplayContractError(f"{label} contains duplicate case_id {case_id}")
        indexed[case_id] = dict(row)
    return indexed


def _paper_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PAPER_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def verify_manifest_hashes(manifest_path: Path, *, base: Path) -> dict[str, Any]:
    """Verify every declared input/output that resolves under *base*."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[dict[str, str]] = []
    checked = 0
    for section in ("inputs", "outputs"):
        records = manifest.get(section, [])
        if not isinstance(records, list):
            failures.append({"path": section, "reason": "not_a_list"})
            continue
        for record in records:
            if not isinstance(record, Mapping):
                failures.append({"path": section, "reason": "record_not_object"})
                continue
            relative = record.get("path")
            expected = record.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                failures.append({"path": str(relative), "reason": "invalid_record"})
                continue
            candidate = (base / relative).resolve()
            try:
                candidate.relative_to(base.resolve())
            except ValueError:
                failures.append({"path": relative, "reason": "path_escapes_base"})
                continue
            if not candidate.is_file():
                failures.append({"path": relative, "reason": "missing"})
                continue
            checked += 1
            if sha256_file(candidate) != expected:
                failures.append({"path": relative, "reason": "sha256_mismatch"})
    return {
        "status": "PASS" if not failures else "FAIL",
        "files_checked": checked,
        "failures": failures,
    }


def load_replay_inputs(source_run: Path) -> list[dict[str, Any]]:
    """Join immutable case, raw-response, and prediction ledgers by case ID."""

    missing = [source_run / name for name in REQUIRED_SOURCE_FILES if not (source_run / name).is_file()]
    if missing:
        raise ReplayContractError(f"missing source artifact: {missing[0]}")

    source_audit = verify_manifest_hashes(source_run / "manifest.json", base=PAPER_ROOT)
    if source_audit["status"] != "PASS":
        first = source_audit["failures"][0]
        raise ReplayContractError(
            f"source manifest integrity failed: {first['path']} ({first['reason']})"
        )

    cases = _index_unique(_read_jsonl(source_run / "cases.jsonl"), "cases")
    raw = _index_unique(_read_jsonl(source_run / "raw_responses.jsonl"), "raw responses")
    predictions = _index_unique(
        _read_jsonl(source_run / "predictions.jsonl"), "predictions"
    )
    key_sets = {frozenset(cases), frozenset(raw), frozenset(predictions)}
    if len(key_sets) != 1:
        raise ReplayContractError("case/raw/prediction ledgers do not have identical case IDs")

    joined: list[dict[str, Any]] = []
    for case_id in sorted(cases):
        case = cases[case_id]
        raw_row = raw[case_id]
        prediction = predictions[case_id]
        text = raw_row.get("text")
        if not isinstance(text, str):
            raise ReplayContractError(f"raw response {case_id} has no string text")
        observed_text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        expected_text_hash = prediction.get("raw_text_sha256")
        if observed_text_hash != expected_text_hash:
            raise ReplayContractError(f"raw text hash mismatch for {case_id}")
        if raw_row.get("raw_response_sha256") != prediction.get("raw_response_sha256"):
            raise ReplayContractError(f"transport-response hash mismatch for {case_id}")
        joined.append(
            {
                "case": case,
                "raw": raw_row,
                "prediction": prediction,
                "raw_text_sha256": observed_text_hash,
            }
        )
    return joined


def classify_extraction_stages(extraction: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the typed extraction result into an explicit stage ledger."""

    status = ExtractionStatus(str(extraction.get("status")))
    diagnostics = [str(item) for item in extraction.get("diagnostics", ())]
    valid_empty = status is ExtractionStatus.EMPTY and any(
        "valid response contains no entities or relations" in item
        for item in diagnostics
    )
    response_empty = status is ExtractionStatus.EMPTY and not valid_empty

    parse_passed = status not in {
        ExtractionStatus.MALFORMED,
        ExtractionStatus.CLIENT_ERROR,
    } and not response_empty
    schema_passed = parse_passed and status not in {
        ExtractionStatus.SCHEMA_INVALID,
        ExtractionStatus.PARTIAL,
    }
    canonicalization_completed = schema_passed
    nonempty_graph = status is ExtractionStatus.SUCCESS
    return {
        "json_parse": {
            "status": "passed" if parse_passed else "failed",
        },
        "schema_validation": {
            "status": "passed" if schema_passed else "failed",
        },
        "canonicalization": {
            "status": (
                "passed"
                if nonempty_graph
                else "completed_empty"
                if canonicalization_completed
                else "blocked"
            ),
            "entity_count": len(extraction.get("entities", ())),
            "relation_count": len(extraction.get("relations", ())),
        },
        "nonempty_extraction_gate": {
            "status": "passed" if nonempty_graph else "failed",
            "extraction_status": status.value,
        },
    }


def _stage_passed(row: Mapping[str, Any], stage: str) -> bool:
    return row.get("stages", {}).get(stage, {}).get("status") == "passed"


def compute_attrition(ledger: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute exact, observation-level pipeline attrition."""

    stage_counts = {
        "raw_hash_verified": sum(_stage_passed(row, "source_integrity") for row in ledger),
        "json_parse_passed": sum(_stage_passed(row, "json_parse") for row in ledger),
        "schema_passed": sum(_stage_passed(row, "schema_validation") for row in ledger),
        "canonicalization_completed": sum(
            row.get("stages", {}).get("canonicalization", {}).get("status")
            in {"passed", "completed_empty"}
            for row in ledger
        ),
        "nonempty_canonical_graph": sum(
            _stage_passed(row, "nonempty_extraction_gate") for row in ledger
        ),
        "ontology_load_validated": sum(_stage_passed(row, "ontology_load") for row in ledger),
        "pellet_consistent": sum(_stage_passed(row, "pellet") for row in ledger),
        "inference_ledgers_emitted": sum(
            _stage_passed(row, "inference_ledger") for row in ledger
        ),
        "serialized_verified_artifacts": sum(
            _stage_passed(row, "serialization") for row in ledger
        ),
    }
    total = len(ledger)
    extraction_status = Counter(
        row.get("stages", {})
        .get("nonempty_extraction_gate", {})
        .get("extraction_status", "missing")
        for row in ledger
    )
    reasoning_status = Counter(
        row.get("stages", {}).get("pellet", {}).get("reasoning_status", "not_run")
        for row in ledger
    )
    inferred_count = sum(
        int(row.get("stages", {}).get("inference_ledger", {}).get("inferred_relation_count", 0))
        for row in ledger
    )
    inferred_with_proofs = sum(
        int(row.get("stages", {}).get("inference_ledger", {}).get("relations_with_proofs", 0))
        for row in ledger
    )
    by_condition: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ledger:
        grouped[str(row.get("condition"))].append(row)
    for condition, rows in sorted(grouped.items()):
        by_condition[condition] = {
            "case_count": len(rows),
            "extraction_status_distribution": dict(
                sorted(
                    Counter(
                        row["stages"]["nonempty_extraction_gate"]["extraction_status"]
                        for row in rows
                    ).items()
                )
            ),
            "serialized_verified_artifacts": sum(
                _stage_passed(row, "serialization") for row in rows
            ),
        }
    return {
        "case_count": total,
        "stage_counts": stage_counts,
        "stage_rates": {
            key: (value / total if total else None) for key, value in stage_counts.items()
        },
        "extraction_status_distribution": dict(sorted(extraction_status.items())),
        "reasoning_status_distribution": dict(sorted(reasoning_status.items())),
        "cases_with_inferred_relations": sum(
            int(row.get("stages", {}).get("inference_ledger", {}).get("inferred_relation_count", 0)) > 0
            for row in ledger
        ),
        "inferred_relation_count": inferred_count,
        "inferred_relations_with_proofs": inferred_with_proofs,
        "by_condition": by_condition,
    }


def build_integrity_gate(
    ledger: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any], *, expected_case_count: int = 126
) -> dict[str, Any]:
    """Apply deterministic G6 checks to the replay evidence."""

    total = len(ledger)
    counts = metrics["stage_counts"]
    failed: list[str] = []
    if total != expected_case_count:
        failed.append(f"exactly_{expected_case_count}_cases")
    if counts["raw_hash_verified"] != total:
        failed.append("all_raw_hashes_verified")
    if any(int(row.get("replay_client_calls", -1)) != 1 for row in ledger):
        failed.append("exactly_one_replay_read_per_case")
    if metrics.get("hosted_call_attempts") != 0:
        failed.append("zero_hosted_calls")
    for row in ledger:
        consistent = _stage_passed(row, "pellet")
        serialized = _stage_passed(row, "serialization")
        if consistent != serialized:
            failed.append("serialization_iff_pellet_consistent")
            break
    if metrics["inferred_relation_count"] != metrics["inferred_relations_with_proofs"]:
        failed.append("all_inferred_relations_have_proof_ledgers")
    return {
        "gate_id": "G6_hosted_end_to_end_replay_integrity",
        "stage": "results_to_manuscript_claims",
        "hypothesis_or_mechanism": (
            "Every stored response is replayed exactly once without repair; only "
            "Pellet-consistent graphs serialize, and every inferred edge has support."
        ),
        "input_artifacts": ["stage_ledger.jsonl", "metrics.json"],
        "command_or_check": "deterministic build_integrity_gate over case-level ledger",
        "pass_criteria": {
            "exact_case_count": expected_case_count,
            "raw_hash_verified": f"{expected_case_count}/{expected_case_count}",
            "hosted_call_attempts": 0,
            "serialization": "iff Pellet status is consistent",
            "proof_coverage": "100% of inferred relations",
        },
        "metrics": {
            "case_count": total,
            "raw_hash_verified": counts["raw_hash_verified"],
            "hosted_call_attempts": metrics.get("hosted_call_attempts"),
            "pellet_consistent": counts["pellet_consistent"],
            "serialized_verified_artifacts": counts["serialized_verified_artifacts"],
            "inferred_relation_count": metrics["inferred_relation_count"],
            "inferred_relations_with_proofs": metrics["inferred_relations_with_proofs"],
        },
        "failed_criteria": sorted(set(failed)),
        "verdict": "PASS" if not failed else "FAIL_REPAIR",
        "fail_interpretation": (
            "The replay is incomplete, untraceable, or violates the fail-closed "
            "serialization/proof contract."
        ),
        "next_allowed_step": "bounded integration reporting",
    }


def run_standard_result_gate(ledger_path: Path, output_path: Path, *, expected_case_count: int = 126) -> dict[str, Any]:
    """Run the repository's standard G6 checker on per-case runtimes."""

    count_str = str(expected_case_count)
    command = [
        sys.executable,
        str(STANDARD_GATE_SCRIPT),
        "--file",
        str(ledger_path),
        "--metric-column",
        "runtime_seconds",
        "--gate-id",
        "G6_hosted_end_to_end_replay_runtime_integrity",
        "--stage",
        "offline_pipeline_replay",
        "--next-allowed-step",
        "bounded_integration_reporting",
        "--fail-interpretation",
        "Per-case replay runtimes are missing, non-finite, or outside the declared execution bound.",
        "--min-rows",
        count_str,
        "--metric-min",
        "0",
        "--metric-max",
        "120",
        "--min-values-per-metric",
        count_str,
        "--out",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ReplayContractError(
            "standard G6 result-integrity gate failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    if report.get("verdict") not in {"PASS", "PASS_WITH_WARNINGS"}:
        raise ReplayContractError(
            f"standard G6 emitted nonpassing verdict: {report.get('verdict')}"
        )
    return report


def _chunk_for_case(case: Mapping[str, Any]) -> StructuredChunk:
    context = case.get("context")
    if not isinstance(context, str):
        raise ReplayContractError(f"case {case.get('case_id')} has no context string")
    words = context.split()
    return StructuredChunk.create(
        text=context,
        source_id=(
            f"ctihal:{case.get('source_path')}#row={case.get('source_row_index')}"
        ),
        index=0,
        start_word=0,
        end_word=len(words),
        document_sha256=str(case.get("source_sha256")),
    )


async def replay_one(
    joined: Mapping[str, Any], *, artifacts_root: Path, logs_root: Path
) -> dict[str, Any]:
    """Replay one frozen response through an isolated production pipeline."""

    case = joined["case"]
    raw = joined["raw"]
    case_id = str(case["case_id"])
    replay_client = ReplayClient(str(raw["text"]))
    extractor = MalOntExtractor(replay_client)
    reasoner = MalOntReasoner(
        ontology_iri=f"http://example.org/malont/replay/{case_id}#"
    )
    case_output = artifacts_root / case_id
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        outcome = await run_pipeline(
            [_chunk_for_case(case)],
            output_dir=case_output,
            extractor=extractor,
            reasoner=reasoner,
        )
    elapsed = time.perf_counter() - started

    extraction = outcome["extraction"]
    stages = {
        "source_integrity": {"status": "passed"},
        **classify_extraction_stages(extraction),
    }
    load = outcome.get("load")
    if load is None:
        stages["ontology_load"] = {
            "status": "blocked",
            "reasoning_status": ReasoningStatus.NOT_RUN.value,
        }
    else:
        stages["ontology_load"] = {
            "status": (
                "passed"
                if load.status is ReasoningStatus.VALIDATED
                else "failed"
            ),
            "reasoning_status": load.status.value,
            "entity_count": load.get("entity_count", 0),
            "relation_count": load.get("relation_count", 0),
        }
    reasoning = outcome.get("reasoning")
    if reasoning is None:
        stages["pellet"] = {
            "status": "blocked",
            "reasoning_status": ReasoningStatus.NOT_RUN.value,
        }
        stages["inference_ledger"] = {
            "status": "blocked",
            "inferred_relation_count": 0,
            "relations_with_proofs": 0,
            "inferred_relations": [],
        }
    else:
        consistent = reasoning.status is ReasoningStatus.CONSISTENT
        inferred = list(reasoning.get("inferred_relations", ()))
        stages["pellet"] = {
            "status": "passed" if consistent else "failed",
            "reasoning_status": reasoning.status.value,
            "entity_count": reasoning.get("entity_count", 0),
            "relation_count": reasoning.get("relation_count", 0),
        }
        stages["inference_ledger"] = {
            "status": "passed" if consistent else "blocked",
            "inferred_relation_count": len(inferred),
            "relations_with_proofs": sum(bool(item.get("proofs")) for item in inferred),
            "inferred_relations": inferred,
        }

    artifact_records: list[dict[str, Any]] = []
    for path_string in outcome.get("artifacts", {}).values():
        path = Path(path_string)
        if path.is_file():
            artifact_records.append(
                {
                    "path": _paper_relative(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    serialized = bool(outcome.get("serialized"))
    stages["serialization"] = {
        "status": "passed" if serialized else "blocked",
        "artifacts": sorted(artifact_records, key=lambda item: item["path"]),
    }

    log_text = "".join(
        (
            "[stdout]\n",
            stdout_buffer.getvalue(),
            "\n[stderr]\n",
            stderr_buffer.getvalue(),
        )
    )
    log_path = logs_root / f"{case_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_text, encoding="utf-8")

    diagnostics = {
        "extraction": list(extraction.get("diagnostics", ())),
        "load": list(load.get("diagnostics", ())) if load is not None else [],
        "reasoning": (
            list(reasoning.get("diagnostics", ())) if reasoning is not None else []
        ),
    }
    return {
        "case_id": case_id,
        "sample_id": case.get("sample_id"),
        "group": case.get("group"),
        "condition": case.get("condition"),
        "repeat": case.get("repeat"),
        "source_path": case.get("source_path"),
        "source_row_index": case.get("source_row_index"),
        "context_sha256": case.get("context_sha256"),
        "raw_text_sha256": joined["raw_text_sha256"],
        "raw_response_sha256": raw.get("raw_response_sha256"),
        "replay_client_calls": replay_client.call_count,
        "hosted_call_attempts": 0,
        "pipeline_status": outcome.get("status"),
        "runtime_seconds": elapsed,
        "stages": stages,
        "diagnostics": diagnostics,
        "log": {
            "path": _paper_relative(log_path),
            "sha256": sha256_file(log_path),
        },
    }


async def execute_replay(
    *, source_run: Path, output_dir: Path, java: str | Path | None = None
) -> dict[str, Any]:
    """Execute the exact offline replay and freeze its evidence artifacts."""

    source_run = source_run.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ReplayContractError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    java_executable, java_audit = resolve_java(java)
    if java_executable is None:
        raise ReplayContractError("a functioning Java executable is required for Pellet")
    owl.JAVA_EXE = str(java_executable)

    joined = load_replay_inputs(source_run)
    started_at = utc_now()
    ledger: list[dict[str, Any]] = []
    artifacts_root = output_dir / "case_artifacts"
    logs_root = output_dir / "logs"
    for item in joined:
        ledger.append(
            await replay_one(item, artifacts_root=artifacts_root, logs_root=logs_root)
        )
    finished_at = utc_now()

    ledger_path = output_dir / "stage_ledger.jsonl"
    failures_path = output_dir / "failures.jsonl"
    metrics_path = output_dir / "metrics.json"
    gate_path = output_dir / "gate_G6_replay_integrity.json"
    standard_gate_path = output_dir / "gate_G6_standard_result_integrity.json"
    config_path = output_dir / "run_config.json"
    write_jsonl(ledger_path, ledger)
    failures = []
    for row in ledger:
        if row["pipeline_status"] != "success":
            stages = row["stages"]
            first_failed = next(
                (
                    name
                    for name in (
                        "json_parse",
                        "schema_validation",
                        "nonempty_extraction_gate",
                        "ontology_load",
                        "pellet",
                        "serialization",
                    )
                    if stages[name]["status"] in {"failed", "blocked"}
                ),
                "unknown",
            )
            failures.append(
                {
                    "case_id": row["case_id"],
                    "condition": row["condition"],
                    "first_nonpassing_stage": first_failed,
                    "pipeline_status": row["pipeline_status"],
                    "diagnostics": row["diagnostics"],
                }
            )
    write_jsonl(failures_path, failures)

    metrics = compute_attrition(ledger)
    metrics.update(
        {
            "schema_version": "1.0",
            "experiment_scope": (
                "offline_frozen_response_pipeline_integration_not_factual_accuracy"
            ),
            "source_run": _paper_relative(source_run),
            "source_run_manifest_sha256": sha256_file(source_run / "manifest.json"),
            "hosted_call_attempts": 0,
            "replay_client_calls": sum(row["replay_client_calls"] for row in ledger),
            "raw_outputs_repaired": False,
            "fixed_response_order": "lexicographic_case_id",
            "total_runtime_seconds": sum(row["runtime_seconds"] for row in ledger),
        }
    )
    write_json(metrics_path, metrics)
    expected_case_count = len(_read_jsonl(source_run / "cases.jsonl"))
    gate = build_integrity_gate(ledger, metrics, expected_case_count=expected_case_count)
    write_json(gate_path, gate)
    if gate["verdict"] != "PASS":
        raise ReplayContractError(
            f"scientific integrity gate failed: {gate['failed_criteria']}"
        )
    standard_gate = run_standard_result_gate(ledger_path, standard_gate_path, expected_case_count=expected_case_count)

    config = {
        "schema_version": "1.0",
        "run_id": output_dir.name,
        "started_at": started_at,
        "finished_at": finished_at,
        "source_run": _paper_relative(source_run),
        "case_order": "lexicographic_case_id",
        "case_count": len(ledger),
        "network_policy": "no hosted client constructed; frozen ReplayClient only",
        "raw_response_policy": "verbatim; no repair, normalization, or retry",
        "java": java_audit,
    }
    write_json(config_path, config)

    counts = metrics["stage_counts"]
    readme = f"""# Hosted-response end-to-end replay

## Scope

This run replays all {metrics['case_count']} frozen response texts from
`{_paper_relative(source_run)}` through the production extractor,
canonicalizer, ontology loader, real Pellet reasoner, inference/proof ledger,
and fail-closed serializer. It makes zero hosted calls and does not repair any
response. CTI-HAL is not MALOnt factual ground truth, so these results measure
pipeline integration attrition, not extraction accuracy.

## Attrition

| Stage | Cases passing/completing |
|---|---:|
| Source/raw hash verified | {counts['raw_hash_verified']} |
| JSON parse | {counts['json_parse_passed']} |
| Schema validation | {counts['schema_passed']} |
| Canonicalization completed (including empty) | {counts['canonicalization_completed']} |
| Non-empty canonical graph | {counts['nonempty_canonical_graph']} |
| Ontology load validated | {counts['ontology_load_validated']} |
| Pellet consistent | {counts['pellet_consistent']} |
| Inference ledger emitted | {counts['inference_ledgers_emitted']} |
| Verified artifact serialized | {counts['serialized_verified_artifacts']} |

Extraction status distribution: `{json.dumps(metrics['extraction_status_distribution'], sort_keys=True)}`.
Inferred relations: {metrics['inferred_relation_count']}; with proof ledgers:
{metrics['inferred_relations_with_proofs']}.

## Reproduction

Run from the paper root:

```bash
python src/run_hosted_end_to_end_replay.py \\
  --source-run reproduced_runs/hosted_stress_20260809T211342Z \\
  --output-dir reproduced_runs/hosted_end_to_end_replay_20260809
```

The runner refuses to overwrite an existing output directory. Hashes for the
source ledgers, runner, production modules, logs, and all serialized artifacts
are recorded in `manifest.json`. Both the replay-specific logical G6 gate and
the repository-standard numeric G6 gate are included in the package.
"""
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")

    source_inputs = [source_run / name for name in REQUIRED_SOURCE_FILES]
    # Resolve module paths explicitly so a manifest can never silently omit
    # the frozen implementation.
    code_inputs = [
        Path(__file__).resolve(),
        PAPER_ROOT / "src" / "main_pipeline.py",
        PAPER_ROOT / "src" / "extraction.py",
        PAPER_ROOT / "src" / "contracts.py",
        PAPER_ROOT / "src" / "reasoning.py",
        PAPER_ROOT / "src" / "experiment_support.py",
    ]
    output_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = build_manifest(
        run_id=output_dir.name,
        command=sys.argv,
        seed=20260809,
        started_at=started_at,
        finished_at=finished_at,
        inputs=source_inputs + code_inputs,
        outputs=output_files,
        base=PAPER_ROOT,
        java=java_audit,
        dependencies=("owlready2",),
        extra={
            "experiment_scope": metrics["experiment_scope"],
            "source_run_manifest_sha256": metrics["source_run_manifest_sha256"],
            "hosted_call_attempts": 0,
            "raw_outputs_repaired": False,
            "scientific_gate": {
                "path": _paper_relative(gate_path),
                "verdict": gate["verdict"],
            },
            "standard_scientific_gate": {
                "path": _paper_relative(standard_gate_path),
                "verdict": standard_gate["verdict"],
            },
        },
    )
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    final_audit = verify_manifest_hashes(manifest_path, base=PAPER_ROOT)
    if final_audit["status"] != "PASS":
        raise ReplayContractError(f"output manifest verification failed: {final_audit}")
    return {
        "metrics": metrics,
        "gate": gate,
        "standard_gate": standard_gate,
        "manifest_audit": final_audit,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay frozen hosted responses through the production pipeline."
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--java", help="optional explicit Java executable")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = asyncio.run(
            execute_replay(
                source_run=args.source_run,
                output_dir=args.output_dir,
                java=args.java,
            )
        )
    except (OSError, ValueError, json.JSONDecodeError, ReplayContractError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

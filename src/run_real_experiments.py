"""Run one fail-closed hosted extraction/reasoning smoke benchmark.

This is a provenance-bearing integration check, not a deployment benchmark and
not evidence for extraction accuracy. Any hosted-call, parsing, validation, or
reasoner failure is recorded as a failed attempt and raises an exception.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # Support both script and package-style execution.
    from .contracts import ExtractionStatus, ReasoningStatus
    from .extraction import MalOntExtractor
    from .llm_client import LLMResponse, MultiModelClient
    from .reasoning import MalOntReasoner
except ImportError:  # pragma: no cover - script entry point.
    from contracts import ExtractionStatus, ReasoningStatus
    from extraction import MalOntExtractor
    from llm_client import LLMResponse, MultiModelClient
    from reasoning import MalOntReasoner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CASE_PATH = PROCESSED_DIR / "real_case_study.json"
BENCHMARK_PATH = PROCESSED_DIR / "live_micro_benchmark.json"

# This controlled sentence tests the ontology signature. It is explicitly
# synthetic and names a CVE as the Vulnerability instead of misclassifying the
# SolarWinds Orion product as one.
SAMPLE_TEXT = (
    "Synthetic evaluation fixture: APT29 uses the SUNBURST malware, and "
    "SUNBURST targets vulnerability CVE-2020-10148 in SolarWinds Orion."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _failure_artifact(
    *,
    started_at: str,
    run_status: str,
    extraction: Any,
    hosted_call: Any,
    diagnostics: list[str],
) -> dict:
    return {
        "artifact_type": "live_integration_smoke_benchmark",
        "evidence_status": "smoke_test_only_not_accuracy_evidence",
        "run_status": run_status,
        "timestamp_started_utc": started_at,
        "timestamp_completed_utc": _utc_now(),
        "input_fixture": {"kind": "synthetic", "text": SAMPLE_TEXT},
        "extraction": dict(extraction) if isinstance(extraction, dict) else None,
        "hosted_call": (
            hosted_call.to_dict(include_text=False)
            if isinstance(hosted_call, LLMResponse)
            else None
        ),
        "diagnostics": diagnostics,
    }


async def main() -> dict:
    print("--- Running hosted extraction/reasoning integration smoke test ---")
    print(f"Input fixture: {SAMPLE_TEXT}")
    started_at = _utc_now()

    client = MultiModelClient()
    extractor = MalOntExtractor(client=client)
    extraction_started = time.perf_counter()
    result = await extractor.extract_from_chunk(SAMPLE_TEXT)
    extraction_elapsed = round(time.perf_counter() - extraction_started, 6)
    hosted_call = client.last_response

    if result.status is not ExtractionStatus.SUCCESS:
        artifact = _failure_artifact(
            started_at=started_at,
            run_status="failed_extraction",
            extraction=result,
            hosted_call=hosted_call,
            diagnostics=list(result.get("diagnostics", ())),
        )
        _write_json(BENCHMARK_PATH, artifact)
        raise RuntimeError(f"extraction failed closed with status={result.status.value}")
    if not isinstance(hosted_call, LLMResponse):
        artifact = _failure_artifact(
            started_at=started_at,
            run_status="failed_missing_hosted_call_metadata",
            extraction=result,
            hosted_call=hosted_call,
            diagnostics=["successful extraction lacks hosted-call provenance"],
        )
        _write_json(BENCHMARK_PATH, artifact)
        raise RuntimeError("successful extraction lacks hosted-call provenance")

    _write_json(CASE_PATH, dict(result))

    reasoner = MalOntReasoner()
    load_result = reasoner.load_extracted_data(result)
    if load_result.status is not ReasoningStatus.VALIDATED:
        artifact = _failure_artifact(
            started_at=started_at,
            run_status="failed_reasoning_input_validation",
            extraction=result,
            hosted_call=hosted_call,
            diagnostics=list(load_result.get("diagnostics", ())),
        )
        _write_json(BENCHMARK_PATH, artifact)
        raise RuntimeError(
            f"reasoning input failed validation with status={load_result.status.value}"
        )

    reasoning_started = time.perf_counter()
    reasoning_result = reasoner.verify_and_infer()
    reasoning_elapsed = round(time.perf_counter() - reasoning_started, 6)
    if reasoning_result.status is not ReasoningStatus.CONSISTENT:
        artifact = _failure_artifact(
            started_at=started_at,
            run_status="failed_reasoning",
            extraction=result,
            hosted_call=hosted_call,
            diagnostics=list(reasoning_result.get("diagnostics", ())),
        )
        artifact["reasoning"] = dict(reasoning_result)
        _write_json(BENCHMARK_PATH, artifact)
        raise RuntimeError(
            f"reasoning failed closed with status={reasoning_result.status.value}"
        )

    benchmark = {
        "artifact_type": "live_integration_smoke_benchmark",
        "evidence_status": "smoke_test_only_not_accuracy_evidence",
        "run_status": "success",
        "timestamp_started_utc": started_at,
        "timestamp_completed_utc": _utc_now(),
        "input_fixture": {"kind": "synthetic", "text": SAMPLE_TEXT},
        "hosted_call": hosted_call.to_dict(include_text=False),
        "latency_seconds": {
            "hosted_call": hosted_call.latency_seconds,
            "validated_extraction_total": extraction_elapsed,
            "owl_swrl_reasoning": reasoning_elapsed,
            "total_extraction_plus_reasoning": round(
                extraction_elapsed + reasoning_elapsed, 6
            ),
        },
        "extraction": {
            "status": result.status.value,
            "entity_count": len(result["entities"]),
            "relation_count": len(result["relations"]),
        },
        "reasoning": dict(reasoning_result),
        "notes": [
            "One synthetic integration fixture; not an accuracy, throughput, or deployment benchmark.",
            "Successful status requires both validated extraction and a completed consistent Pellet run.",
        ],
    }
    _write_json(BENCHMARK_PATH, benchmark)
    print(f"Smoke benchmark artifact written to {BENCHMARK_PATH}")
    return benchmark


if __name__ == "__main__":
    asyncio.run(main())

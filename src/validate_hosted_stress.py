"""Validate a completed hosted-stress run and emit a G5 gate record."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


MODEL_ID = "qwen/qwen3.7-plus"
CONDITIONS = {"minimal_json", "typed_evidence", "typed_evidence_injection"}
ALLOWED_STATUSES = {
    "valid",
    "malformed_json",
    "schema_invalid",
    "evidence_invalid",
    "provider_error",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected an object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"expected an object at {path}:{number}")
        rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    paper_root = run_dir.parents[3]
    required = (
        "cases.jsonl",
        "raw_responses.jsonl",
        "predictions.jsonl",
        "failures.jsonl",
        "metrics.json",
        "run_config.json",
        "pricing_snapshot.json",
        "credential_scan.json",
        "manifest.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing hosted artifacts: {missing}")

    cases = read_jsonl(run_dir / "cases.jsonl")
    raw = read_jsonl(run_dir / "raw_responses.jsonl")
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    config = read_json(run_dir / "run_config.json")
    metrics = read_json(run_dir / "metrics.json")
    credential = read_json(run_dir / "credential_scan.json")
    manifest = read_json(run_dir / "manifest.json")

    failures: list[str] = []
    warnings: list[str] = []
    case_ids = [str(row.get("case_id")) for row in cases]
    prediction_ids = [str(row.get("case_id")) for row in predictions]
    if len(case_ids) != len(set(case_ids)):
        failures.append("case IDs are not unique")
    if len(prediction_ids) != len(set(prediction_ids)):
        failures.append("prediction case IDs are not unique")
    if set(case_ids) != set(prediction_ids):
        failures.append("prediction ledger is not a one-to-one completion of the case ledger")
    if config.get("state") != "completed":
        failures.append("run_config state is not completed")
    if sha256_file(run_dir / "cases.jsonl") != config.get("case_ledger_sha256"):
        failures.append("case ledger hash differs from preregistered hash")
    if metrics.get("case_count_preregistered") != len(cases):
        failures.append("metrics preregistered count does not match cases")
    if metrics.get("case_count_completed") != len(predictions):
        failures.append("metrics completed count does not match predictions")

    condition_counts = Counter(str(row.get("condition")) for row in cases)
    if set(condition_counts) != CONDITIONS or len(set(condition_counts.values())) != 1:
        failures.append("conditions are absent or unbalanced")
    grouped: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in cases:
        grouped[(str(row.get("sample_id")), str(row.get("condition")))].add(
            int(row.get("repeat"))
        )
    expected_repeats = set(range(1, int(config.get("repeats", 0)) + 1))
    if not grouped or any(repeats != expected_repeats for repeats in grouped.values()):
        failures.append("sample-condition repeat groups are incomplete")

    invalid_statuses = sorted(
        {str(row.get("status")) for row in predictions} - ALLOWED_STATUSES
    )
    if invalid_statuses:
        failures.append(f"unexpected prediction statuses: {invalid_statuses}")
    if any(
        bool(row.get("attack_success"))
        for row in predictions
        if row.get("condition") != "typed_evidence_injection"
    ):
        failures.append("attack success was recorded outside the injection condition")

    successful = [row for row in predictions if row.get("status") != "provider_error"]
    raw_ids = {str(row.get("case_id")) for row in raw}
    if raw_ids != {str(row.get("case_id")) for row in successful}:
        failures.append("raw-response ledger does not match successful provider calls")
    for row in successful:
        if not str(row.get("provider", "")).startswith("openrouter_key_"):
            failures.append("a successful row used a non-OpenRouter provider")
            break
        if row.get("requested_model") != MODEL_ID:
            failures.append("a successful row used a non-preregistered model slug")
            break
        if not row.get("resolved_model") or not row.get("raw_response_sha256"):
            failures.append("a successful row lacks resolved-model or raw-response provenance")
            break

    dataset_root = Path(str(config.get("dataset_root", "")))
    for row in cases:
        source = dataset_root / str(row.get("source_path"))
        if not source.is_file() or sha256_file(source) != row.get("source_sha256"):
            failures.append(f"source provenance mismatch for {row.get('case_id')}")
            break

    if credential.get("status") != "PASS" or credential.get("known_secret_values_found") != 0:
        failures.append("credential scan did not pass")

    hash_mismatches: list[str] = []
    for collection in ("inputs", "outputs"):
        for record in manifest.get(collection, []):
            stored = Path(str(record.get("path")))
            candidates = [stored] if stored.is_absolute() else [paper_root / stored, run_dir / stored]
            resolved = next((path for path in candidates if path.is_file()), None)
            if resolved is None or sha256_file(resolved) != record.get("sha256"):
                hash_mismatches.append(str(stored))
    if hash_mismatches:
        failures.append(f"manifest hash verification failed: {hash_mismatches[:5]}")

    provider_errors = sum(row.get("status") == "provider_error" for row in predictions)
    if provider_errors:
        warnings.append(f"{provider_errors} provider-error outcomes retained and not recoded")
    verdict = "PASS" if not failures else "FAIL"
    return {
        "gate_id": "G5_hosted_robustness_validity",
        "stage": "hosted_response_contract_stress",
        "verdict": verdict,
        "pass_criteria": {
            "case_completion": "one prediction per preregistered case",
            "condition_balance": "equal calls in all three conditions",
            "repeat_integrity": "all sample-condition groups contain every preregistered repeat",
            "provider_model": f"OpenRouter only; requested model {MODEL_ID}",
            "provenance": "source, raw response, and manifest hashes verify",
            "failure_semantics": "provider errors remain explicit",
            "credential_scan": "PASS with zero known secret values",
        },
        "metrics": {
            "cases": len(cases),
            "predictions": len(predictions),
            "raw_responses": len(raw),
            "condition_counts": dict(sorted(condition_counts.items())),
            "provider_errors": provider_errors,
            "attack_success_count": sum(bool(row.get("attack_success")) for row in predictions),
            "manifest_records_checked": sum(
                len(manifest.get(name, [])) for name in ("inputs", "outputs")
            ),
        },
        "failed_criteria": failures,
        "warnings": warnings,
        "fail_interpretation": (
            "The hosted stress artifacts do not support a response-contract, "
            "repeatability, latency, or bounded injection claim."
        ),
        "next_allowed_step": "G6_hosted_result_integrity" if verdict == "PASS" else "repair_hosted_artifacts",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate(args.run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

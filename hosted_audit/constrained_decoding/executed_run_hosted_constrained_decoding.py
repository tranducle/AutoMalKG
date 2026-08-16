#!/usr/bin/env python3
"""Provider-native constrained-decoding comparator for the Round-3 CC0 benchmark.

The reported Round-3 run (``run_hosted_round3_cc0.py``) is frozen at 144 calls
over three prompt conditions and must not be reopened. This runner is a separate,
additive comparator that answers one question that run cannot: what does a
*provider-native* JSON Schema constraint buy, and what does it still not buy?

Two conditions, both issued to the same pinned route with
``response_format={"type": "json_schema", ...}``:

``constrained_minimal``
    The byte-identical minimal system prompt of the reported ``minimal_json``
    condition, plus the provider schema. Isolates a single manipulation against
    the reported 0/48 floor.

``constrained_typed``
    The Round-3 typed-evidence system prompt plus the provider schema, i.e. the
    strongest configuration a practitioner could assemble without the contract.

Every response is scored at three nested levels, because the point of the
comparison is the gap between them:

1. ``schema_conformant`` - satisfies the JSON Schema handed to the provider.
2. ``contract_structural`` - also passes the AutoMalKG typed contract, which
   additionally requires unique entity IDs, no dangling relation endpoints, and
   permitted subject/object types per relation signature. None of those three is
   expressible in JSON Schema.
3. ``contract_full`` - also passes the evidence-substring rule.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_support import (  # noqa: E402
    DEFAULT_RUNS_ROOT,
    build_manifest,
    sha256_file,
    utc_now,
    utc_run_stamp,
    write_json,
    write_jsonl,
)
from llm_client import AllProvidersFailedError, MultiModelClient  # noqa: E402
from run_hosted_round3_cc0 import (  # noqa: E402
    DEFAULT_BENCHMARK_DIR,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_SEED,
    EXPECTED_CONTEXT_COUNT,
    ROUND3_TYPED_SYSTEM_PROMPT,
    _evidence_diagnostics,
    load_benchmark,
    score_against_gold,
)
from run_hosted_stress import (  # noqa: E402
    MINIMAL_SYSTEM_PROMPT,
    MODEL_ID,
    PROVIDER_FILTER,
    cost_from_usage,
    fetch_pricing_snapshot,
    parse_json_response,
    quantile,
    stable_hash,
)
from extraction import validate_extraction_schema  # noqa: E402

CONDITIONS = ("constrained_minimal", "constrained_typed")
DEFAULT_REPEATS = 3
DEFAULT_BUDGET_USD = 1.0
EXPECTED_CASE_COUNT = 96  # 16 contexts x 2 conditions x 3 repeats

ENTITY_TYPE_ENUM = ["ThreatActor", "Malware", "Vulnerability", "Campaign", "Indicator"]
PREDICATE_ENUM = ["Uses", "Targets", "AssociatedWith", "HasIndicator"]

#: The strongest schema the AutoMalKG contract can be expressed as in JSON
#: Schema. Everything the contract checks beyond this - ID uniqueness,
#: referential integrity, signature typing, evidence provenance - has no JSON
#: Schema encoding, which is precisely what this comparator measures.
CONSTRAINED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["entities", "relations"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "type", "name", "evidence"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string", "enum": ENTITY_TYPE_ENUM},
                    "name": {"type": "string"},
                    "evidence": {"type": "string"},
                },
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["subject", "predicate", "object", "evidence"],
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string", "enum": PREDICATE_ENUM},
                    "object": {"type": "string"},
                    "evidence": {"type": "string"},
                },
            },
        },
    },
}

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "automalkg_extraction",
        "strict": True,
        "schema": CONSTRAINED_SCHEMA,
    },
}


class ConstrainedRunConfigurationError(RuntimeError):
    """Raised when execution would violate this comparator's frozen contract."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def provider_schema_diagnostics(payload: Any) -> list[str]:
    """Check a payload against ``CONSTRAINED_SCHEMA`` without a schema library.

    The schema is fixed and shallow, so a direct check is more auditable here
    than pulling in a validator dependency the rest of the artifact does not use.
    """

    diagnostics: list[str] = []
    if not isinstance(payload, Mapping):
        return ["root must be a JSON object"]
    unknown = sorted(set(payload) - {"entities", "relations"})
    diagnostics.extend(f"root has unknown key: {key}" for key in unknown)
    for key in ("entities", "relations"):
        if key not in payload:
            diagnostics.append(f"root is missing required key: {key}")

    specs = (
        ("entities", ("id", "type", "name", "evidence"), "type", ENTITY_TYPE_ENUM),
        (
            "relations",
            ("subject", "predicate", "object", "evidence"),
            "predicate",
            PREDICATE_ENUM,
        ),
    )
    for collection, required, enum_field, permitted in specs:
        rows = payload.get(collection)
        if not isinstance(rows, list):
            if collection in payload:
                diagnostics.append(f"'{collection}' must be a list")
            continue
        for index, row in enumerate(rows):
            location = f"{collection}[{index}]"
            if not isinstance(row, Mapping):
                diagnostics.append(f"{location} must be an object")
                continue
            for key in sorted(set(row) - set(required)):
                diagnostics.append(f"{location} has unknown key: {key}")
            for field in required:
                if field not in row:
                    diagnostics.append(f"{location} is missing required field: {field}")
                elif not isinstance(row[field], str):
                    diagnostics.append(f"{location}.{field} must be a string")
            value = row.get(enum_field)
            if isinstance(value, str) and value not in permitted:
                diagnostics.append(
                    f"{location}.{enum_field} is outside the declared enum: {value!r}"
                )
    return diagnostics


def evaluate_constrained_response(case: Mapping[str, Any], text: str) -> dict[str, Any]:
    """Score one response at the three nested acceptance levels."""

    payload, parse_error, transport = parse_json_response(text)
    if parse_error:
        return {
            "status": "malformed_json",
            "transport": transport,
            "schema_conformant": False,
            "contract_structural": False,
            "contract_full": False,
            "provider_schema_diagnostics": [parse_error],
            "diagnostics": [parse_error],
            "evidence_diagnostics": [],
            "prediction": None,
            "prediction_sha256": None,
        }

    schema_diagnostics = provider_schema_diagnostics(payload)
    _, _, contract_diagnostics = validate_extraction_schema(payload)
    evidence_diagnostics = _evidence_diagnostics(payload, str(case["context"]))
    if contract_diagnostics:
        status = "schema_invalid"
    elif evidence_diagnostics:
        status = "evidence_invalid"
    else:
        status = "valid"
    return {
        "status": status,
        "transport": transport,
        "schema_conformant": not schema_diagnostics,
        "contract_structural": not contract_diagnostics,
        "contract_full": not contract_diagnostics and not evidence_diagnostics,
        "provider_schema_diagnostics": schema_diagnostics,
        "diagnostics": contract_diagnostics,
        "evidence_diagnostics": evidence_diagnostics,
        "prediction": payload,
        "prediction_sha256": stable_hash(payload),
    }


def _prompt_for(condition: str, context: str) -> tuple[str, str]:
    """Render system+user prompts, reusing the reported run's prompts verbatim."""

    if condition == "constrained_minimal":
        return MINIMAL_SYSTEM_PROMPT, f"<cti_source>\n{context}\n</cti_source>"
    if condition == "constrained_typed":
        return (
            ROUND3_TYPED_SYSTEM_PROMPT,
            f"<cti_source>\n{context}\n</cti_source>\n\n"
            "Evidence may cite only <cti_source>.",
        )
    raise ValueError(f"unknown condition: {condition}")


def generate_cases(
    contexts: Sequence[Mapping[str, Any]], *, repeats: int = DEFAULT_REPEATS
) -> list[dict[str, Any]]:
    """Generate a deterministic condition-alternating execution schedule."""

    if repeats != DEFAULT_REPEATS:
        raise ConstrainedRunConfigurationError("this comparator requires three repeats")
    ordered = sorted(contexts, key=lambda row: int(row["ordinal"]))
    if len(ordered) != EXPECTED_CONTEXT_COUNT:
        raise ConstrainedRunConfigurationError("benchmark must contain 16 contexts")
    cases: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        for slot in range(len(CONDITIONS)):
            for position, sample_row in enumerate(ordered):
                sample = dict(sample_row)
                base_index = int(sample["ordinal"]) - 1
                condition = CONDITIONS[(slot + base_index + repeat - 1) % len(CONDITIONS)]
                system_prompt, user_prompt = _prompt_for(condition, str(sample["text"]))
                case = {
                    **sample,
                    "context": sample["text"],
                    "case_id": f"{sample['sample_id']}__{condition}__r{repeat}",
                    "execution_index": len(cases) + 1,
                    "repeat": repeat,
                    "condition": condition,
                    "condition_ordinal_within_context_repeat": slot + 1,
                    "system_prompt": system_prompt,
                    "system_prompt_sha256": hashlib.sha256(
                        system_prompt.encode("utf-8")
                    ).hexdigest(),
                    "user_prompt": user_prompt,
                    "user_prompt_sha256": hashlib.sha256(
                        user_prompt.encode("utf-8")
                    ).hexdigest(),
                    "response_format_sha256": stable_hash(RESPONSE_FORMAT),
                    "requested_model": MODEL_ID,
                    "provider_filter": PROVIDER_FILTER,
                    "temperature": 0.0,
                }
                case["case_contract_sha256"] = stable_hash(case)
                cases.append(case)
    if len(cases) != EXPECTED_CASE_COUNT:
        raise ConstrainedRunConfigurationError(
            f"schedule must hold {EXPECTED_CASE_COUNT} cases, got {len(cases)}"
        )
    seen = Counter(str(row["case_id"]) for row in cases)
    if any(count != 1 for count in seen.values()):
        raise ConstrainedRunConfigurationError("duplicate case id in schedule")
    return cases


def compute_metrics(
    cases: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    pricing: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate acceptance levels, repeat stability, latency, tokens, and cost."""

    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_condition[str(row["condition"])].append(row)

    condition_metrics: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = by_condition.get(condition, [])
        latencies = [
            float(row["latency_seconds"])
            for row in rows
            if row.get("latency_seconds") is not None
        ]
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["sample_id"])].append(row)
        expected_groups = {
            str(case["sample_id"]) for case in cases if case["condition"] == condition
        }
        exact_raw = 0
        exact_prediction = 0
        for sample_id in expected_groups:
            repeats = grouped.get(sample_id, [])
            expected_repeats = len(
                [
                    case
                    for case in cases
                    if case["condition"] == condition and case["sample_id"] == sample_id
                ]
            )
            if len(repeats) != expected_repeats:
                continue
            raw_hashes = [row.get("raw_text_sha256") for row in repeats]
            pred_hashes = [row.get("prediction_sha256") for row in repeats]
            if None not in raw_hashes and len(set(raw_hashes)) == 1:
                exact_raw += 1
            if None not in pred_hashes and len(set(pred_hashes)) == 1:
                exact_prediction += 1
        denominator = len(expected_groups)
        total = len(rows)
        condition_metrics[condition] = {
            "completed_calls": total,
            "status_distribution": dict(
                sorted(Counter(str(row["status"]) for row in rows).items())
            ),
            "schema_conformant_count": sum(bool(row["schema_conformant"]) for row in rows),
            "schema_conformant_rate": (
                sum(bool(row["schema_conformant"]) for row in rows) / total if total else None
            ),
            "contract_structural_count": sum(bool(row["contract_structural"]) for row in rows),
            "contract_structural_rate": (
                sum(bool(row["contract_structural"]) for row in rows) / total if total else None
            ),
            "contract_full_count": sum(bool(row["contract_full"]) for row in rows),
            "valid_rate": (
                sum(bool(row["contract_full"]) for row in rows) / total if total else None
            ),
            "contract_rejection_reasons": dict(
                sorted(
                    Counter(
                        diagnostic.split(":")[0]
                        for row in rows
                        for diagnostic in row.get("diagnostics", [])
                    ).items()
                )
            ),
            "repeat_groups": denominator,
            "exact_raw_repeat_stability_rate": exact_raw / denominator if denominator else None,
            "exact_prediction_repeat_stability_rate": (
                exact_prediction / denominator if denominator else None
            ),
            "latency_seconds": {
                "median": quantile(latencies, 0.5),
                "iqr": (
                    quantile(latencies, 0.75) - quantile(latencies, 0.25)
                    if latencies
                    else None
                ),
                "p95": quantile(latencies, 0.95),
            },
        }

    token_totals: Counter[str] = Counter()
    known_cost = 0.0
    missing_usage = 0
    for row in predictions:
        usage = row.get("token_usage") or {}
        for key, value in usage.items():
            if isinstance(value, int):
                token_totals[key] += value
        cost = cost_from_usage(usage, pricing)
        if cost is None:
            missing_usage += 1
        else:
            known_cost += cost
    return {
        "conditions": condition_metrics,
        "case_count_preregistered": len(cases),
        "case_count_completed": len(predictions),
        "calls_missing_token_usage": missing_usage,
        "token_totals": dict(sorted(token_totals.items())),
        "known_cost_usd": known_cost,
        "constrained_decoding": {
            "response_format": RESPONSE_FORMAT,
            "response_format_sha256": stable_hash(RESPONSE_FORMAT),
        },
    }


def _prompt_byte_envelope(case: Mapping[str, Any]) -> int:
    return len(str(case["system_prompt"]).encode("utf-8")) + len(
        str(case["user_prompt"]).encode("utf-8")
    )


def _worst_case_cost(case: Mapping[str, Any], pricing: Mapping[str, Any], cap: int) -> float:
    rates = pricing["pricing_per_token_usd"]
    return _prompt_byte_envelope(case) * float(rates["prompt"]) + cap * float(
        rates["completion"]
    )


def _single_route_client() -> tuple[MultiModelClient, list[dict[str, str]]]:
    os.environ["LLM_PROVIDER_FILTER"] = PROVIDER_FILTER
    os.environ["OPENROUTER_MODEL"] = MODEL_ID
    client = MultiModelClient()
    providers = [
        {"provider": str(row.get("name")), "requested_model": str(row.get("model"))}
        for row in client.providers
    ]
    if not providers:
        raise ConstrainedRunConfigurationError("no OpenRouter provider is configured")
    if any(not row["provider"].startswith("openrouter_key_") for row in providers):
        raise ConstrainedRunConfigurationError("provider filter did not isolate OpenRouter")
    if any(row["requested_model"] != MODEL_ID for row in providers):
        raise ConstrainedRunConfigurationError("configured model differs from the frozen slug")
    client.providers = client.providers[:1]
    return client, providers[:1]


async def execute(run_dir: Path, *, budget_usd: float, cap: int) -> dict[str, Any]:
    """Issue the scheduled calls sequentially, checkpointing after every one."""

    cases = _read_jsonl(run_dir / "cases.jsonl")
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    if sha256_file(run_dir / "cases.jsonl") != config.get("case_ledger_sha256"):
        raise ConstrainedRunConfigurationError("case ledger changed after preparation")
    pricing = json.loads((run_dir / "pricing_snapshot.json").read_text(encoding="utf-8"))
    client, providers = _single_route_client()

    raw_path = run_dir / "raw_responses.jsonl"
    prediction_path = run_dir / "predictions.jsonl"
    failure_path = run_dir / "failures.jsonl"
    raw_rows = _read_jsonl(raw_path) if raw_path.is_file() else []
    predictions = _read_jsonl(prediction_path) if prediction_path.is_file() else []
    failures = _read_jsonl(failure_path) if failure_path.is_file() else []
    completed = {str(row["case_id"]) for row in predictions}
    accounted_cost = sum(
        cost_from_usage(row.get("token_usage") or {}, pricing) or 0.0 for row in predictions
    )

    config["state"] = "executing_partial_outputs_checkpointed"
    config["started_at"] = utc_now()
    config["provider_contract"] = providers
    write_json(run_dir / "run_config.json", config)

    for case in cases:
        if str(case["case_id"]) in completed:
            continue
        worst_case = _worst_case_cost(case, pricing, cap)
        if accounted_cost + worst_case > budget_usd:
            failures.append({
                "case_id": case["case_id"],
                "stage": "budget_stop",
                "error_type": "ConservativeBudgetGuard",
                "message": "accounted cost plus the next request upper bound exceeds the budget",
            })
            write_jsonl(failure_path, failures)
            break
        try:
            response = await client.generate_response(
                str(case["system_prompt"]),
                str(case["user_prompt"]),
                temperature=float(case["temperature"]),
                max_completion_tokens=cap,
                response_format=RESPONSE_FORMAT,
            )
        except AllProvidersFailedError as exc:
            failures.append({"case_id": case["case_id"], "stage": "hosted_call", **exc.to_dict()})
            write_jsonl(failure_path, failures)
            print(json.dumps({"progress": case["execution_index"], "status": "provider_error"}), flush=True)
            break

        usage = dict(response.token_usage)
        observed = usage.get("completion_tokens")
        cap_violation = isinstance(observed, int) and observed > cap
        raw_rows.append({
            "case_id": case["case_id"],
            "execution_index": case["execution_index"],
            "request": {
                "provider_filter": PROVIDER_FILTER,
                "requested_model": MODEL_ID,
                "temperature": case["temperature"],
                "max_completion_tokens": cap,
                "response_format_sha256": case["response_format_sha256"],
                "system_prompt_sha256": case["system_prompt_sha256"],
                "user_prompt_sha256": case["user_prompt_sha256"],
            },
            **response.to_dict(include_text=True),
        })
        evaluation = evaluate_constrained_response(case, response.text)
        prediction = {
            "case_id": case["case_id"],
            "execution_index": case["execution_index"],
            "sample_id": case["sample_id"],
            "group": case["group"],
            "context_sha256": case["context_sha256"],
            "gold_sha256": case["gold_sha256"],
            "condition": case["condition"],
            "repeat": case["repeat"],
            "provider": response.provider,
            "requested_model": response.requested_model,
            "resolved_model": response.resolved_model,
            "request_id": response.request_id,
            "request_parameters": {
                "temperature": case["temperature"],
                "max_completion_tokens": cap,
                "response_format_sha256": case["response_format_sha256"],
            },
            "prompt_hashes": {
                "system": case["system_prompt_sha256"],
                "user": case["user_prompt_sha256"],
            },
            "timestamp_utc": response.timestamp_utc,
            "latency_seconds": response.latency_seconds,
            "token_usage": usage,
            "raw_response_sha256": response.raw_response_sha256,
            "raw_text_sha256": hashlib.sha256(response.text.encode("utf-8")).hexdigest(),
            "completion_cap_violation": cap_violation,
            "gold_score": score_against_gold(evaluation.get("prediction"), case["gold"]),
            **evaluation,
        }
        predictions.append(prediction)
        call_cost = cost_from_usage(usage, pricing)
        accounted_cost += worst_case if call_cost is None else call_cost
        if not evaluation["contract_full"] or cap_violation:
            failures.append({
                "case_id": case["case_id"],
                "stage": "response_validation",
                "status": evaluation["status"],
                "schema_conformant": evaluation["schema_conformant"],
                "contract_structural": evaluation["contract_structural"],
                "completion_cap_violation": cap_violation,
                "diagnostics": evaluation["diagnostics"] + evaluation["evidence_diagnostics"],
            })
        write_jsonl(raw_path, raw_rows)
        write_jsonl(prediction_path, predictions)
        write_jsonl(failure_path, failures)
        print(json.dumps({
            "progress": case["execution_index"],
            "of": len(cases),
            "condition": case["condition"],
            "status": evaluation["status"],
            "schema_conformant": evaluation["schema_conformant"],
            "accounted_cost_usd": round(accounted_cost, 8),
        }, sort_keys=True), flush=True)
        if cap_violation:
            break

    write_jsonl(raw_path, raw_rows)
    write_jsonl(prediction_path, predictions)
    write_jsonl(failure_path, failures)
    metrics = {
        **compute_metrics(cases, predictions, pricing),
        "benchmark_scope": "author-created CC0 controlled design set",
        "real_world_cti_accuracy_claim_permitted": False,
        "comparator_scope": (
            "provider-native constrained decoding on the same pinned route as the "
            "reported Round-3 run; not a replication of that run"
        ),
        "cost_accounting": {
            "budget_usd": budget_usd,
            "accounted_cost_usd": accounted_cost,
            "provider_billing_may_differ": True,
        },
    }
    write_json(run_dir / "metrics.json", metrics)
    config["state"] = "completed" if len(predictions) == len(cases) else "incomplete"
    config["finished_at"] = utc_now()
    config["case_count_completed"] = len(predictions)
    write_json(run_dir / "run_config.json", config)
    write_json(
        run_dir / "manifest.json",
        build_manifest(
            run_id=run_dir.name,
            command=[Path(__file__).name, "--execute", str(run_dir)],
            seed=DEFAULT_SEED,
            started_at=str(config["started_at"]),
            finished_at=str(config["finished_at"]),
            inputs=[Path(config["benchmark_dir"])],
            outputs=[raw_path, prediction_path, failure_path, run_dir / "metrics.json"],
            base=run_dir.parents[3],
            dependencies=("openai",),
            extra={"conditions": list(CONDITIONS), "constrained_decoding": True},
        ),
    )
    return metrics


def prepare(benchmark_dir: Path, output_root: Path, *, cap: int, budget_usd: float) -> Path:
    """Freeze the schedule, pricing, and configuration before any paid call."""

    contexts, source = load_benchmark(benchmark_dir)
    cases = generate_cases(contexts)
    run_dir = output_root / f"hosted_constrained_decoding_{utc_run_stamp()}"
    if run_dir.exists():
        raise ConstrainedRunConfigurationError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    write_jsonl(run_dir / "cases.jsonl", cases)
    pricing = fetch_pricing_snapshot()
    write_json(run_dir / "pricing_snapshot.json", pricing)
    snapshot = run_dir / "executed_run_hosted_constrained_decoding.py"
    snapshot.write_bytes(Path(__file__).read_bytes())
    projected = sum(_worst_case_cost(case, pricing, cap) for case in cases)
    config = {
        "state": "prepared_no_hosted_calls",
        "benchmark_dir": str(benchmark_dir),
        "benchmark_id": source.get("benchmark_id"),
        "case_count": len(cases),
        "case_ledger_sha256": sha256_file(run_dir / "cases.jsonl"),
        "runner_snapshot_sha256": sha256_file(snapshot),
        "pricing_snapshot_sha256": sha256_file(run_dir / "pricing_snapshot.json"),
        "conditions": list(CONDITIONS),
        "contexts": len(contexts),
        "repeats": DEFAULT_REPEATS,
        "max_completion_tokens": cap,
        "budget_usd": budget_usd,
        "requested_model": MODEL_ID,
        "provider_filter": PROVIDER_FILTER,
        "response_format": RESPONSE_FORMAT,
        "response_format_sha256": stable_hash(RESPONSE_FORMAT),
        "prepared_at": utc_now(),
        "projection": {
            "case_count": len(cases),
            "projected_upper_bound_usd": projected,
            "prompt_token_estimator": (
                "conservative UTF-8 byte-count prompt-token envelope (upper bound)"
            ),
        },
        "claim_boundary": (
            "controlled author-created design-set metrics only; no real-world CTI "
            "accuracy or generalization"
        ),
    }
    write_json(run_dir / "run_config.json", config)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="freeze a schedule; no paid calls")
    parser.add_argument("--execute", type=Path, help="execute a prepared run directory")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument(
        "--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prepare:
        run_dir = prepare(
            args.benchmark_dir,
            args.output_root,
            cap=args.max_completion_tokens,
            budget_usd=args.budget_usd,
        )
        print(json.dumps({"prepared_run_dir": str(run_dir)}, sort_keys=True))
        return 0
    if args.execute:
        metrics = asyncio.run(
            execute(
                args.execute.resolve(),
                budget_usd=args.budget_usd,
                cap=args.max_completion_tokens,
            )
        )
        print(json.dumps(metrics["conditions"], indent=1, sort_keys=True))
        return 0
    build_parser().print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

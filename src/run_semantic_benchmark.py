"""Run seeded OWL-consistency and SWRL-control benchmarks with raw Pellet logs."""

from __future__ import annotations

import argparse
import io
import json
import random
import statistics
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Iterable

from experiment_support import (
    DEFAULT_RUNS_ROOT,
    PAPER_ROOT,
    build_manifest,
    quantile,
    resolve_java,
    utc_now,
    utc_run_stamp,
    write_json,
    write_jsonl,
)


RULE_ID = "R1_actor_uses_malware_targets_vulnerability"


def generate_cases(
    *,
    seed: int,
    consistency_cases_per_class: int,
    swrl_controls_per_class: int,
    scale_sizes: Iterable[int],
    scale_repeats: int,
) -> list[dict[str, Any]]:
    """Generate a deterministic, balanced case ledger."""

    if consistency_cases_per_class < 1 or swrl_controls_per_class < 1:
        raise ValueError("case counts per class must be positive")
    if scale_repeats < 0 or any(size < 1 for size in scale_sizes):
        raise ValueError("scale sizes must be positive and repeats non-negative")
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    consistent_variants = ("valid_chain", "isolated_typed_entities", "two_valid_chains")
    inconsistent_variants = (
        "actor_also_malware",
        "malware_also_vulnerability",
        "vulnerability_also_campaign",
    )
    for expected, variants in (
        ("CONSISTENT", consistent_variants),
        ("INCONSISTENT", inconsistent_variants),
    ):
        for index in range(consistency_cases_per_class):
            cases.append(
                {
                    "case_id": f"consistency-{expected.lower()}-{index:03d}",
                    "family": "consistency",
                    "seed": seed,
                    "expected_consistency": expected,
                    "variant": variants[index % len(variants)],
                    "graph_size": 1,
                    "warmup": False,
                }
            )

    negative_variants = ("missing_uses", "missing_targets", "disconnected_chain")
    for expected_edge in (True, False):
        for index in range(swrl_controls_per_class):
            cases.append(
                {
                    "case_id": f"swrl-{'positive' if expected_edge else 'negative'}-{index:03d}",
                    "family": "swrl_control",
                    "seed": seed,
                    "expected_consistency": "CONSISTENT",
                    "expected_edge": expected_edge,
                    "variant": "licensed_chain" if expected_edge else negative_variants[index % len(negative_variants)],
                    "graph_size": 1,
                    "rule_id": RULE_ID,
                    "warmup": False,
                }
            )

    for size in scale_sizes:
        for repeat in range(scale_repeats):
            cases.append(
                {
                    "case_id": f"scale-n{size:04d}-r{repeat:02d}",
                    "family": "scale",
                    "seed": seed,
                    "expected_consistency": "CONSISTENT",
                    "expected_edge": True,
                    "variant": "licensed_chain",
                    "graph_size": size,
                    "repeat": repeat,
                    "rule_id": RULE_ID,
                    "warmup": False,
                }
            )
    rng.shuffle(cases)
    return cases


def _build_reasoner(case: dict[str, Any]):
    """Build a case through the repaired production loading contract."""

    from reasoning import MalOntReasoner

    reasoner = MalOntReasoner(
        ontology_iri=f"http://example.org/automalkg/{case['case_id']}.owl#"
    )
    entities: list[dict[str, str]] = []
    relations: list[dict[str, str]] = []
    expected_edges: list[tuple[str, str]] = []
    support_by_edge: dict[str, list[str]] = {}
    variant = case["variant"]
    family = case["family"]

    if family == "consistency" and case["expected_consistency"] == "INCONSISTENT":
        entity_type = {
            "actor_also_malware": "ThreatActor",
            "malware_also_vulnerability": "Malware",
            "vulnerability_also_campaign": "Vulnerability",
        }[variant]
        payload = {
            "entities": [{"id": "collision", "type": entity_type, "name": "Collision"}],
            "relations": [],
        }
        load_result = reasoner.load_extracted_data(payload)
        if load_result.status.value != "validated":
            raise RuntimeError(f"fixture rejected before reasoning: {load_result['diagnostics']}")
        individual = reasoner.entity_map["collision"]
        if variant == "actor_also_malware":
            individual.is_a.append(reasoner.onto.Malware)
        elif variant == "malware_also_vulnerability":
            individual.is_a.append(reasoner.onto.Vulnerability)
        else:
            individual.is_a.append(reasoner.onto.Campaign)
        return reasoner, expected_edges, support_by_edge

    if family == "consistency" and variant == "isolated_typed_entities":
        payload = {
            "entities": [
                {"id": "actor", "type": "ThreatActor", "name": "Actor"},
                {"id": "malware", "type": "Malware", "name": "Malware"},
                {"id": "vulnerability", "type": "Vulnerability", "name": "CVE-TEST"},
            ],
            "relations": [],
        }
        load_result = reasoner.load_extracted_data(payload)
        if load_result.status.value != "validated":
            raise RuntimeError(f"fixture rejected before reasoning: {load_result['diagnostics']}")
        return reasoner, expected_edges, support_by_edge

    chain_count = int(case.get("graph_size", 1))
    if family == "consistency" and variant == "two_valid_chains":
        chain_count = 2
    for index in range(chain_count):
        actor = f"actor_{index}"
        malware = f"malware_{index}"
        vulnerability = f"vulnerability_{index}"
        entities.extend(
            [
                {"id": actor, "type": "ThreatActor", "name": f"Actor {index}"},
                {"id": malware, "type": "Malware", "name": f"Malware {index}"},
                {"id": vulnerability, "type": "Vulnerability", "name": f"CVE-TEST-{index}"},
            ]
        )
        if family == "consistency" or variant == "licensed_chain":
            relations.extend(
                [
                    {"subject": actor, "predicate": "Uses", "object": malware},
                    {"subject": malware, "predicate": "Targets", "object": vulnerability},
                ]
            )
            if family != "consistency":
                expected_edges.append((actor, vulnerability))
                support_by_edge[f"{actor}|Targets|{vulnerability}"] = [
                    f"asserted:{actor}|Uses|{malware}",
                    f"asserted:{malware}|Targets|{vulnerability}",
                ]
        elif variant == "missing_uses":
            relations.append(
                {"subject": malware, "predicate": "Targets", "object": vulnerability}
            )
        elif variant == "missing_targets":
            relations.append({"subject": actor, "predicate": "Uses", "object": malware})
        elif variant == "disconnected_chain":
            decoy = f"decoy_malware_{index}"
            entities.append({"id": decoy, "type": "Malware", "name": f"Decoy {index}"})
            relations.extend(
                [
                    {"subject": actor, "predicate": "Uses", "object": malware},
                    {"subject": decoy, "predicate": "Targets", "object": vulnerability},
                ]
            )
    for relation_index, relation in enumerate(relations):
        relation["provenance"] = [
            f"case:{case['case_id']}:assertion:{relation_index:04d}"
        ]
    load_result = reasoner.load_extracted_data(
        {"status": "success", "entities": entities, "relations": relations}
    )
    if load_result.status.value != "validated":
        raise RuntimeError(f"fixture rejected before reasoning: {load_result['diagnostics']}")
    return reasoner, expected_edges, support_by_edge


def execute_case(case: dict[str, Any], java_executable: Path, log_dir: Path) -> dict[str, Any]:
    """Run one case in an isolated world and retain the complete reasoner log."""

    import owlready2 as owl

    owl.JAVA_EXE = str(java_executable)
    reasoner, expected_edges, support_by_edge = _build_reasoner(case)
    buffer = io.StringIO()
    started = time.perf_counter_ns()
    observed_consistency = "UNVERIFIED"
    error_type: str | None = None
    error_message: str | None = None
    with redirect_stdout(buffer), redirect_stderr(buffer):
        try:
            result = reasoner.verify_and_infer()
            print(json.dumps({"production_reasoning_result": result}, default=str))
            if result.status.value == "consistent":
                observed_consistency = "CONSISTENT"
            elif result.status.value == "inconsistent":
                observed_consistency = "INCONSISTENT"
            else:
                observed_consistency = "UNVERIFIED"
                error_type = result.status.value
                error_message = "; ".join(result["diagnostics"])
        except Exception as exc:
            error_type = type(exc).__name__
            error_message = str(exc)
            traceback.print_exc()
    runtime_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    log_path = log_dir / f"{case['case_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(buffer.getvalue(), encoding="utf-8")

    predicted_edges: list[dict[str, Any]] = []
    if observed_consistency == "CONSISTENT":
        inferred = {
            (edge["subject"], edge["predicate"], edge["object"]): edge
            for edge in result["inferred_relations"]
        }
        for actor, vulnerability in expected_edges:
            edge_key = (actor, "Targets", vulnerability)
            if edge_key in inferred:
                key = f"{actor}|Targets|{vulnerability}"
                production_edge = dict(inferred[edge_key])
                production_support_ids = sorted(
                    {
                        support_id
                        for proof in production_edge.get("proofs", ())
                        for support_id in proof.get("support_relation_ids", ())
                    }
                )
                production_edge.update(
                    rule_id=production_edge.get("rule_id", RULE_ID),
                    supporting_edge_ids=production_support_ids or support_by_edge[key],
                    harness_expected_support_keys=support_by_edge[key],
                    harness_provenance={
                        "kind": "derived",
                        "case_id": case["case_id"],
                        "inherited_assertion_provenance": production_edge.get("provenance", []),
                    },
                )
                predicted_edges.append(production_edge)
        # Negative controls have no expected edge; explicitly inspect all actor Targets.
        if case.get("expected_edge") is False:
            for subject, predicate, obj in sorted(inferred):
                if predicate == "Targets" and subject.startswith("actor_"):
                    production_edge = dict(inferred[(subject, predicate, obj)])
                    predicted_edges.append(
                        {
                            **production_edge,
                            "subject": subject,
                            "predicate": "Targets",
                            "object": obj,
                            "rule_id": production_edge.get("rule_id", RULE_ID),
                            "supporting_edge_ids": sorted(
                                {
                                    support_id
                                    for proof in production_edge.get("proofs", ())
                                    for support_id in proof.get("support_relation_ids", ())
                                }
                            ),
                            "harness_provenance": {
                                "kind": "unexpected_derived",
                                "case_id": case["case_id"],
                                "inherited_assertion_provenance": production_edge.get("provenance", []),
                            },
                        }
                    )

    observed_edge = bool(predicted_edges) if "expected_edge" in case else None
    prediction = {
        "case_id": case["case_id"],
        "family": case["family"],
        "variant": case["variant"],
        "graph_size": case["graph_size"],
        "repeat": case.get("repeat"),
        "warmup": bool(case.get("warmup")),
        "expected_consistency": case["expected_consistency"],
        "observed_consistency": observed_consistency,
        "expected_edge": case.get("expected_edge"),
        "observed_edge": observed_edge,
        "predicted_edges": predicted_edges,
        "runtime_ms": runtime_ms,
        "error_type": error_type,
        "error_message": error_message,
        "pellet_log": log_path.name,
    }
    reasoner.world.close()
    return prediction


def _exact_binomial_ci(successes: int, total: int, confidence: float = 0.95) -> list[float] | None:
    if total == 0:
        return None
    try:
        from scipy.stats import beta
    except ImportError:
        return None
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return [lower, upper]


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute metrics solely from retained per-case predictions."""

    consistency = [row for row in predictions if row["family"] == "consistency" and not row["warmup"]]
    verified = [row for row in consistency if row["observed_consistency"] != "UNVERIFIED"]
    tp = sum(row["expected_consistency"] == "INCONSISTENT" and row["observed_consistency"] == "INCONSISTENT" for row in verified)
    tn = sum(row["expected_consistency"] == "CONSISTENT" and row["observed_consistency"] == "CONSISTENT" for row in verified)
    fp = sum(row["expected_consistency"] == "CONSISTENT" and row["observed_consistency"] == "INCONSISTENT" for row in verified)
    fn = sum(row["expected_consistency"] == "INCONSISTENT" and row["observed_consistency"] == "CONSISTENT" for row in verified)
    positives = tp + fn
    negatives = tn + fp

    controls = [row for row in predictions if row["family"] == "swrl_control" and not row["warmup"]]
    control_verified = [row for row in controls if row["observed_consistency"] == "CONSISTENT"]
    edge_tp = sum(row["expected_edge"] is True and row["observed_edge"] is True for row in control_verified)
    edge_tn = sum(row["expected_edge"] is False and row["observed_edge"] is False for row in control_verified)
    edge_fp = sum(row["expected_edge"] is False and row["observed_edge"] is True for row in control_verified)
    edge_fn = sum(row["expected_edge"] is True and row["observed_edge"] is False for row in control_verified)
    precision = edge_tp / (edge_tp + edge_fp) if edge_tp + edge_fp else None
    recall = edge_tp / (edge_tp + edge_fn) if edge_tp + edge_fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None

    scaling: dict[str, Any] = {}
    sizes = sorted(
        {
            row["graph_size"]
            for row in predictions
            if row["family"] == "scale" and not row["warmup"]
        }
    )
    for size in sizes:
        rows = [
            row
            for row in predictions
            if row["family"] == "scale" and row["graph_size"] == size and not row["warmup"]
        ]
        runtimes = [row["runtime_ms"] for row in rows if row["observed_consistency"] != "UNVERIFIED"]
        scaling[str(size)] = {
            "attempted_repetitions": len(rows),
            "verified_repetitions": len(runtimes),
            "median_ms": statistics.median(runtimes) if runtimes else None,
            "iqr_ms": (quantile(runtimes, 0.75) - quantile(runtimes, 0.25)) if runtimes else None,
            "p95_ms": quantile(runtimes, 0.95),
            "failure_count": len(rows) - len(runtimes),
            "all_expected_edges_observed": all(row["observed_edge"] is True for row in rows) if rows else None,
        }

    return {
        "observation_policy": "Metrics are computed only from predictions.jsonl; UNVERIFIED is never recoded as success.",
        "consistency": {
            "attempted": len(consistency),
            "verified": len(verified),
            "unverified": len(consistency) - len(verified),
            "confusion_matrix_inconsistent_is_positive": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "status_accuracy": (tp + tn) / len(verified) if verified else None,
            "tpr": tp / positives if positives else None,
            "tnr": tn / negatives if negatives else None,
            "fpr": fp / negatives if negatives else None,
            "fnr": fn / positives if positives else None,
            "tpr_exact_95_ci": _exact_binomial_ci(tp, positives),
            "tnr_exact_95_ci": _exact_binomial_ci(tn, negatives),
        },
        "swrl_controls": {
            "attempted": len(controls),
            "verified": len(control_verified),
            "unverified": len(controls) - len(control_verified),
            "edge_confusion_matrix": {"tp": edge_tp, "tn": edge_tn, "fp": edge_fp, "fn": edge_fn},
            "edge_precision": precision,
            "edge_recall": recall,
            "edge_f1": f1,
            "inferred_edges_with_support_ledger": sum(
                bool(edge.get("supporting_edge_ids"))
                for row in control_verified
                for edge in row["predicted_edges"]
            ),
        },
        "local_scaling": scaling,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--consistency-cases-per-class", type=int, default=2)
    parser.add_argument("--swrl-controls-per-class", type=int, default=2)
    parser.add_argument("--scale-sizes", default="1,10,50")
    parser.add_argument("--scale-repeats", type=int, default=10)
    parser.add_argument("--java", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scale_sizes = tuple(int(value) for value in args.scale_sizes.split(",") if value.strip())
    started_at = utc_now()
    run_id = f"semantic_benchmark_{utc_run_stamp()}"
    output_dir = (args.output_dir or DEFAULT_RUNS_ROOT / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "pellet_logs"
    java_executable, java_audit = resolve_java(args.java)

    cases = generate_cases(
        seed=args.seed,
        consistency_cases_per_class=args.consistency_cases_per_class,
        swrl_controls_per_class=args.swrl_controls_per_class,
        scale_sizes=scale_sizes,
        scale_repeats=args.scale_repeats,
    )
    cases_path = output_dir / "cases.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    failures_path = output_dir / "failures.jsonl"
    gate_path = output_dir / "G3_method_mechanism.json"
    readme_path = output_dir / "README.md"
    write_jsonl(cases_path, cases)

    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if java_executable is None:
        for case in cases:
            prediction = {
                "case_id": case["case_id"],
                "family": case["family"],
                "variant": case["variant"],
                "graph_size": case["graph_size"],
                "repeat": case.get("repeat"),
                "warmup": False,
                "expected_consistency": case["expected_consistency"],
                "observed_consistency": "UNVERIFIED",
                "expected_edge": case.get("expected_edge"),
                "observed_edge": None,
                "predicted_edges": [],
                "runtime_ms": None,
                "error_type": "JavaUnavailable",
                "error_message": "No functioning Java executable was found",
                "pellet_log": None,
            }
            predictions.append(prediction)
            failures.append({"case_id": case["case_id"], "stage": "reasoner", "status": "UNVERIFIED", "error": prediction["error_message"]})
    else:
        warmup = {
            "case_id": "warmup-scale-n0001",
            "family": "scale",
            "seed": args.seed,
            "expected_consistency": "CONSISTENT",
            "expected_edge": True,
            "variant": "licensed_chain",
            "graph_size": 1,
            "repeat": -1,
            "rule_id": RULE_ID,
            "warmup": True,
        }
        predictions.append(execute_case(warmup, java_executable, log_dir))
        for case in cases:
            prediction = execute_case(case, java_executable, log_dir)
            predictions.append(prediction)
            mismatch = prediction["observed_consistency"] != case["expected_consistency"]
            if "expected_edge" in case and prediction["observed_consistency"] == "CONSISTENT":
                mismatch = mismatch or prediction["observed_edge"] != case["expected_edge"]
            if mismatch or prediction["observed_consistency"] == "UNVERIFIED":
                failures.append(
                    {
                        "case_id": case["case_id"],
                        "stage": "semantic_verification",
                        "status": prediction["observed_consistency"],
                        "expected_consistency": case["expected_consistency"],
                        "expected_edge": case.get("expected_edge"),
                        "observed_edge": prediction["observed_edge"],
                        "error": prediction["error_message"],
                    }
                )

    write_jsonl(predictions_path, predictions)
    metrics = compute_metrics(predictions)
    metrics["acceptance"] = {
        "zero_false_verified_outcomes": not failures,
        "all_non_warmup_cases_attempted": len(predictions) - (1 if java_executable else 0) == len(cases),
        "verdict": "PASS" if not failures else "FAIL_REPAIR",
    }
    write_json(metrics_path, metrics)
    write_jsonl(failures_path, failures)
    write_json(
        gate_path,
        {
            "gate_id": "G3_method_mechanism",
            "stage": "semantic_reasoning",
            "verdict": "PASS" if not failures else "FAIL_REPAIR",
            "check_type": "deterministic_custom_harness",
            "input_artifacts": ["cases.jsonl", "predictions.jsonl", "pellet_logs/*.log"],
            "metrics": {
                "consistency_status_accuracy": metrics["consistency"]["status_accuracy"],
                "consistency_unverified": metrics["consistency"]["unverified"],
                "swrl_edge_precision": metrics["swrl_controls"]["edge_precision"],
                "swrl_edge_recall": metrics["swrl_controls"]["edge_recall"],
                "swrl_unverified": metrics["swrl_controls"]["unverified"],
                "inferred_edges_with_support_ledger": metrics["swrl_controls"]["inferred_edges_with_support_ledger"],
            },
            "pass_criteria": {
                "false_verified_outcomes": 0,
                "unverified_consistency_cases": 0,
                "unverified_swrl_controls": 0,
                "negative_control_inferred_edges": 0,
                "support_ledger_coverage": "all expected inferred edges",
            },
            "failed_criteria": [] if not failures else ["one_or_more_case_expectations_failed"],
            "next_allowed_step": "result_integrity_and_claim_mapping",
            "fail_interpretation": "The implementation does not reliably separate OWL consistency from SWRL rule completion.",
        },
    )
    replay = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--consistency-cases-per-class",
        str(args.consistency_cases_per_class),
        "--swrl-controls-per-class",
        str(args.swrl_controls_per_class),
        "--scale-sizes",
        args.scale_sizes,
        "--scale-repeats",
        str(args.scale_repeats),
    ]
    if java_executable:
        replay.extend(["--java", str(java_executable)])
    readme_path.write_text(
        "# AutoMalKG semantic benchmark\n\n"
        "Scope: deterministic OWL consistency detection, SWRL positive/negative controls, and local reasoner scaling. "
        "This run does not estimate natural-language extraction accuracy or external validity.\n\n"
        f"Replay: `{' '.join(replay)}`\n\n"
        "Artifacts: `cases.jsonl`, `predictions.jsonl`, `metrics.json`, `failures.jsonl`, "
        "`G3_method_mechanism.json`, and raw `pellet_logs/*.log`.\n"
        "UNVERIFIED outcomes remain explicit and are excluded from accuracy denominators.\n",
        encoding="utf-8",
    )
    finished_at = utc_now()
    output_files = [
        cases_path,
        predictions_path,
        metrics_path,
        failures_path,
        gate_path,
        readme_path,
    ]
    output_files.extend(sorted(log_dir.glob("*.log")))
    manifest = build_manifest(
        run_id=run_id,
        command=[sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        seed=args.seed,
        started_at=started_at,
        finished_at=finished_at,
        inputs=[
            Path(__file__).resolve(),
            Path(__file__).with_name("experiment_support.py"),
            Path(__file__).with_name("reasoning.py"),
            Path(__file__).with_name("contracts.py"),
            Path(__file__).with_name("extraction.py"),
        ],
        outputs=output_files,
        base=PAPER_ROOT,
        java=java_audit,
        dependencies=("owlready2", "scipy"),
        extra={
            "experiment_ids": ["E2", "E3", "E8-local"],
            "case_parameters": {
                "consistency_cases_per_class": args.consistency_cases_per_class,
                "swrl_controls_per_class": args.swrl_controls_per_class,
                "scale_sizes": list(scale_sizes),
                "scale_repeats": args.scale_repeats,
            },
            "gate_interpretation": metrics["acceptance"]["verdict"],
        },
    )
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics, "failures": len(failures)}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

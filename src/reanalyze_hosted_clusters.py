"""Offline, context-clustered reanalysis of a frozen hosted stress run.

This script performs no network or model calls.  It treats the sampled CTI
contexts, rather than repeated calls, as the analysis clusters.  Its semantic
set comparison is a response-difference diagnostic, not a factual-accuracy
assessment; CTI-HAL does not provide full MALOnt entity/relation gold labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiment_support import (
    DEFAULT_RUNS_ROOT,
    PAPER_ROOT,
    build_manifest,
    quantile,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)


CONDITIONS = ("minimal_json", "typed_evidence", "typed_evidence_injection")
DEFAULT_SOURCE_RUN = DEFAULT_RUNS_ROOT / "hosted_stress_20260809T211342Z"
DEFAULT_SEMANTIC_RUN = DEFAULT_RUNS_ROOT / "semantic_benchmark_seed20260809_comprehensive"
DEFAULT_OUTPUT = DEFAULT_RUNS_ROOT / "hosted_clustered_reanalysis_20260809"
DEFAULT_SEED = 20260809
DEFAULT_BOOTSTRAP_REPLICATES = 10_000


class ReanalysisContractError(RuntimeError):
    """Raised when frozen inputs cannot support the declared paired analysis."""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _normal(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _stable_text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _set_summary(clean: set[str], injected: set[str]) -> dict[str, Any]:
    union = clean | injected
    return {
        "clean_count": len(clean),
        "injected_count": len(injected),
        "retained_count": len(clean & injected),
        "additions": sorted(injected - clean),
        "deletions": sorted(clean - injected),
        "jaccard": 1.0 if not union else len(clean & injected) / len(union),
    }


def canonical_prediction(payload: Any) -> dict[str, Any]:
    """Return ID-insensitive semantic and evidence sets for a parsed response.

    Comparability only requires structurally readable string fields.  It does
    not imply that a response passed the MALOnt schema validator.
    """

    if not isinstance(payload, Mapping):
        return {"comparable": False, "reason": "prediction is not an object"}
    entities = payload.get("entities")
    relations = payload.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        return {"comparable": False, "reason": "entities/relations are not arrays"}

    entity_set: set[str] = set()
    id_to_entity: dict[str, str] = {}
    evidence_records: set[str] = set()
    for index, row in enumerate(entities):
        if not isinstance(row, Mapping):
            return {"comparable": False, "reason": f"entities[{index}] is not an object"}
        entity_id, entity_type, name = row.get("id"), row.get("type"), row.get("name")
        if not all(isinstance(value, str) for value in (entity_id, entity_type, name)):
            return {"comparable": False, "reason": f"entities[{index}] lacks string id/type/name"}
        key = f"entity|{_normal(entity_type)}|{_normal(name)}"
        entity_set.add(key)
        id_to_entity.setdefault(entity_id, key)
        evidence = row.get("evidence")
        if isinstance(evidence, str):
            evidence_records.add(f"{key}|evidence_sha256:{_stable_text_hash(evidence)}")

    relation_set: set[str] = set()
    relation_owners: list[tuple[str, Mapping[str, Any]]] = []
    for index, row in enumerate(relations):
        if not isinstance(row, Mapping):
            return {"comparable": False, "reason": f"relations[{index}] is not an object"}
        subject, predicate, obj = row.get("subject"), row.get("predicate"), row.get("object")
        if not all(isinstance(value, str) for value in (subject, predicate, obj)):
            return {
                "comparable": False,
                "reason": f"relations[{index}] lacks string subject/predicate/object",
            }
        subject_key = id_to_entity.get(subject, f"unresolved_id|{_normal(subject)}")
        object_key = id_to_entity.get(obj, f"unresolved_id|{_normal(obj)}")
        key = f"relation|{subject_key}|{_normal(predicate)}|{object_key}"
        relation_set.add(key)
        relation_owners.append((key, row))
    for key, row in relation_owners:
        evidence = row.get("evidence")
        if isinstance(evidence, str):
            evidence_records.add(f"{key}|evidence_sha256:{_stable_text_hash(evidence)}")

    combined = {f"E:{value}" for value in entity_set} | {
        f"R:{value}" for value in relation_set
    }
    return {
        "comparable": True,
        "entity_set": entity_set,
        "relation_set": relation_set,
        "combined_set": combined,
        "evidence_records": evidence_records,
    }


def paired_cluster_bootstrap(
    context_rows: Sequence[Mapping[str, Any]],
    minuend: str,
    subtrahend: str,
    *,
    seed: int = DEFAULT_SEED,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Bootstrap a paired mean difference by resampling whole contexts."""

    if not context_rows:
        raise ValueError("at least one context cluster is required")
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    differences = [
        float(row["conditions"][minuend]["valid_fraction"])
        - float(row["conditions"][subtrahend]["valid_fraction"])
        for row in context_rows
    ]
    rng = random.Random(seed)
    n = len(differences)
    samples = [
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(replicates)
    ]
    return {
        "estimand": f"mean_context_valid_fraction({minuend}) - mean_context_valid_fraction({subtrahend})",
        "cluster_unit": "sampled CTI context",
        "cluster_count": n,
        "point_difference": sum(differences) / n,
        "percentile_95_interval": [quantile(samples, 0.025), quantile(samples, 0.975)],
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "interpretation": (
            "Deterministic cluster-bootstrap sensitivity interval over the 14 observed "
            "contexts; not a population confidence interval or causal effect."
        ),
    }


def verify_frozen_source(source_run: Path, required: Iterable[Path]) -> dict[str, Any]:
    manifest = read_json(source_run / "manifest.json")
    records = {
        str(record.get("path")): str(record.get("sha256"))
        for collection in ("inputs", "outputs")
        for record in manifest.get(collection, [])
    }
    failures: list[dict[str, str]] = []
    checked: list[dict[str, str]] = []
    for path in required:
        try:
            relative = path.resolve().relative_to(PAPER_ROOT.resolve()).as_posix()
        except ValueError:
            relative = str(path.resolve())
        expected = records.get(relative)
        actual = sha256_file(path) if path.is_file() else None
        if expected is None:
            failures.append({"path": relative, "reason": "absent_from_source_manifest"})
        elif actual != expected:
            failures.append({"path": relative, "reason": "sha256_mismatch"})
        else:
            checked.append({"path": relative, "sha256": actual})
    return {"status": "PASS" if not failures else "FAIL", "checked": checked, "failures": failures}


def verify_output_manifest(manifest_path: Path, *, base: Path = PAPER_ROOT) -> dict[str, Any]:
    """Verify every non-self manifest input/output hash after materialization."""

    manifest = read_json(manifest_path)
    failures: list[dict[str, str]] = []
    checked = 0
    for collection in ("inputs", "outputs"):
        for record in manifest.get(collection, []):
            stored = Path(str(record.get("path")))
            path = stored if stored.is_absolute() else base / stored
            if not path.is_file():
                failures.append({"path": str(stored), "reason": "missing"})
            elif sha256_file(path) != record.get("sha256"):
                failures.append({"path": str(stored), "reason": "sha256_mismatch"})
            else:
                checked += 1
    return {
        "status": "PASS" if not failures else "FAIL",
        "manifest": manifest_path.name,
        "records_checked": checked,
        "failures": failures,
        "scope_note": "manifest.json and this verification report are excluded to avoid recursive hashes",
    }


def build_context_outcomes(
    cases: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    prediction_by_case = {str(row["case_id"]): row for row in predictions}
    by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        by_sample[str(case["sample_id"])].append(case)

    outcomes: list[dict[str, Any]] = []
    for sample_id, sample_cases in sorted(by_sample.items()):
        exemplar = sample_cases[0]
        context = str(exemplar["context"])
        condition_values: dict[str, Any] = {}
        for condition in CONDITIONS:
            relevant = sorted(
                (row for row in sample_cases if row["condition"] == condition),
                key=lambda row: int(row["repeat"]),
            )
            statuses = [str(prediction_by_case[str(row["case_id"])]["status"]) for row in relevant]
            valid = sum(status == "valid" for status in statuses)
            condition_values[condition] = {
                "repeat_count": len(relevant),
                "valid_count": valid,
                "valid_fraction": valid / len(relevant) if relevant else None,
                "statuses_by_repeat": statuses,
            }
        outcomes.append(
            {
                "sample_id": sample_id,
                "context_sha256": exemplar["context_sha256"],
                "group": exemplar["group"],
                "report": Path(str(exemplar["source_path"])).stem,
                "source_path": exemplar["source_path"],
                "source_row_index": exemplar["source_row_index"],
                "context_characters": len(context),
                "context_words": len(re.findall(r"\b\w+\b", context, flags=re.UNICODE)),
                "schema_expressibility_proxy": None,
                "conditions": condition_values,
            }
        )
    return outcomes


def build_injection_pairs(
    cases: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    raw_responses: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    case_index = {str(row["case_id"]): row for row in cases}
    raw_index = {str(row["case_id"]): row for row in raw_responses}
    indexed = {
        (str(row["sample_id"]), str(row["condition"]), int(row["repeat"])): row
        for row in predictions
    }
    samples = sorted({str(row["sample_id"]) for row in predictions})
    repeats = sorted({int(row["repeat"]) for row in predictions})
    pairs: list[dict[str, Any]] = []
    for sample_id in samples:
        for repeat in repeats:
            clean = indexed[(sample_id, "typed_evidence", repeat)]
            injected = indexed[(sample_id, "typed_evidence_injection", repeat)]
            clean_canonical = canonical_prediction(clean.get("prediction"))
            injected_canonical = canonical_prediction(injected.get("prediction"))
            comparable = bool(clean_canonical["comparable"] and injected_canonical["comparable"])
            canary = str(case_index[str(injected["case_id"])].get("attack_canary") or "")
            injected_raw = str(raw_index.get(str(injected["case_id"]), {}).get("text") or "")
            injected_prediction_json = json.dumps(
                injected.get("prediction"), sort_keys=True, ensure_ascii=False
            )
            record: dict[str, Any] = {
                "sample_id": sample_id,
                "group": clean["group"],
                "context_sha256": clean["context_sha256"],
                "repeat": repeat,
                "clean_case_id": clean["case_id"],
                "injected_case_id": injected["case_id"],
                "clean_status": clean["status"],
                "injected_status": injected["status"],
                "schema_status_transition": f"{clean['status']}->{injected['status']}",
                "clean_prediction_sha256": clean.get("prediction_sha256"),
                "injected_prediction_sha256": injected.get("prediction_sha256"),
                "exact_prediction_equal": clean.get("prediction_sha256")
                == injected.get("prediction_sha256"),
                "comparable_semantic_sets": comparable,
                "comparison_reason": None if comparable else {
                    "clean": clean_canonical.get("reason"),
                    "injected": injected_canonical.get("reason"),
                },
                "canary": {
                    "attack_success_under_frozen_definition": bool(injected.get("attack_success")),
                    "present_in_injected_raw_text": bool(canary and canary in injected_raw),
                    "present_in_injected_prediction": bool(canary and canary in injected_prediction_json),
                },
            }
            if comparable:
                record["entity_sets"] = _set_summary(
                    clean_canonical["entity_set"], injected_canonical["entity_set"]
                )
                record["relation_sets"] = _set_summary(
                    clean_canonical["relation_set"], injected_canonical["relation_set"]
                )
                record["combined_semantic_sets"] = _set_summary(
                    clean_canonical["combined_set"], injected_canonical["combined_set"]
                )
                record["evidence_string_hash_sets"] = _set_summary(
                    clean_canonical["evidence_records"], injected_canonical["evidence_records"]
                )
                record["semantic_output_changed"] = (
                    record["combined_semantic_sets"]["jaccard"] < 1.0
                )
                record["evidence_output_changed"] = (
                    record["evidence_string_hash_sets"]["jaccard"] < 1.0
                )
                record["clean_nonempty_to_injected_empty"] = bool(
                    clean_canonical["combined_set"] and not injected_canonical["combined_set"]
                )
                record["clean_empty_to_injected_nonempty"] = bool(
                    not clean_canonical["combined_set"] and injected_canonical["combined_set"]
                )
            pairs.append(record)
    return pairs


def semantic_template_inventory(semantic_cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    family_variants: dict[str, set[str]] = defaultdict(set)
    for row in semantic_cases:
        family_variants[str(row["family"])].add(str(row["variant"]))
    labels = {variant for variants in family_variants.values() for variant in variants}
    return {
        "source": "frozen cases generated by src/run_semantic_benchmark.py",
        "family_scoped_template_count": sum(len(values) for values in family_variants.values()),
        "unique_variant_label_count": len(labels),
        "variants_by_family": {
            family: sorted(variants) for family, variants in sorted(family_variants.items())
        },
        "scale_graph_sizes": sorted(
            {int(row["graph_size"]) for row in semantic_cases if row["family"] == "scale"}
        ),
        "counting_note": (
            "The licensed_chain label occurs in both SWRL-control and scale families, "
            "so family-scoped templates (11) and unique variant labels (10) differ."
        ),
    }


def compute_metrics(
    context_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    semantic_templates: Mapping[str, Any],
    *,
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    condition_metrics: dict[str, Any] = {}
    for condition in CONDITIONS:
        fractions = [float(row["conditions"][condition]["valid_fraction"]) for row in context_rows]
        status_counts: Counter[str] = Counter()
        for row in context_rows:
            status_counts.update(row["conditions"][condition]["statuses_by_repeat"])
        condition_metrics[condition] = {
            "context_count": len(fractions),
            "call_count": sum(row["conditions"][condition]["repeat_count"] for row in context_rows),
            "valid_calls": sum(row["conditions"][condition]["valid_count"] for row in context_rows),
            "mean_context_valid_fraction": sum(fractions) / len(fractions),
            "context_valid_fraction_distribution": dict(
                sorted(Counter(str(value) for value in fractions).items())
            ),
            "status_distribution": dict(sorted(status_counts.items())),
        }

    comparisons = {
        "typed_evidence_minus_minimal_json": paired_cluster_bootstrap(
            context_rows,
            "typed_evidence",
            "minimal_json",
            seed=seed,
            replicates=bootstrap_replicates,
        ),
        "typed_evidence_injection_minus_minimal_json": paired_cluster_bootstrap(
            context_rows,
            "typed_evidence_injection",
            "minimal_json",
            seed=seed + 1,
            replicates=bootstrap_replicates,
        ),
        "typed_evidence_injection_minus_typed_evidence": paired_cluster_bootstrap(
            context_rows,
            "typed_evidence_injection",
            "typed_evidence",
            seed=seed + 2,
            replicates=bootstrap_replicates,
        ),
    }

    comparable = [row for row in pair_rows if row["comparable_semantic_sets"]]
    transitions = Counter(str(row["schema_status_transition"]) for row in pair_rows)
    def count_true(name: str) -> int:
        return sum(bool(row.get(name)) for row in comparable)

    def mean_nested(section: str, key: str) -> float | None:
        values = [float(row[section][key]) for row in comparable]
        return sum(values) / len(values) if values else None

    def set_change_summary(section: str) -> dict[str, Any]:
        jaccards = [float(row[section]["jaccard"]) for row in comparable]
        return {
            "total_additions_across_pairs": sum(
                len(row[section]["additions"]) for row in comparable
            ),
            "total_deletions_across_pairs": sum(
                len(row[section]["deletions"]) for row in comparable
            ),
            "jaccard_distribution": dict(
                sorted(Counter(str(value) for value in jaccards).items())
            ),
        }

    sample_chars = [int(row["context_characters"]) for row in context_rows]
    sample_words = [int(row["context_words"]) for row in context_rows]
    groups = Counter(str(row["group"]) for row in context_rows)
    reports = Counter(str(row["report"]) for row in context_rows)
    return {
        "analysis_scope": "frozen hosted response-contract outputs; not extraction accuracy or factual correctness",
        "analysis_unit": "14 sampled CTI contexts; three repeats are nested within each context",
        "condition_order": list(CONDITIONS),
        "condition_design_warning": (
            "Conditions ran in a fixed, non-counterbalanced order. The minimal-to-typed contrast "
            "bundles ontology enums, relation signatures, evidence constraints, empty-output guidance, "
            "and untrusted-data instructions. Prompt components and temporal order are confounded; "
            "paired differences are descriptive and not component-level causal effects."
        ),
        "conditions": condition_metrics,
        "paired_context_differences": comparisons,
        "injection_pair_analysis": {
            "aligned_pairs": len(pair_rows),
            "comparable_semantic_pairs": len(comparable),
            "schema_status_transitions": dict(sorted(transitions.items())),
            "exact_prediction_equal_pairs": sum(bool(row["exact_prediction_equal"]) for row in pair_rows),
            "any_exact_prediction_displacement_pairs": sum(
                not bool(row["exact_prediction_equal"]) for row in pair_rows
            ),
            "semantic_output_changed_pairs": count_true("semantic_output_changed"),
            "evidence_output_changed_pairs": count_true("evidence_output_changed"),
            "clean_nonempty_to_injected_empty_pairs": count_true("clean_nonempty_to_injected_empty"),
            "clean_empty_to_injected_nonempty_pairs": count_true("clean_empty_to_injected_nonempty"),
            "mean_entity_set_jaccard": mean_nested("entity_sets", "jaccard"),
            "mean_relation_set_jaccard": mean_nested("relation_sets", "jaccard"),
            "mean_combined_semantic_set_jaccard": mean_nested("combined_semantic_sets", "jaccard"),
            "mean_evidence_hash_set_jaccard": mean_nested("evidence_string_hash_sets", "jaccard"),
            "entity_set_changes": set_change_summary("entity_sets"),
            "relation_set_changes": set_change_summary("relation_sets"),
            "combined_semantic_set_changes": set_change_summary("combined_semantic_sets"),
            "evidence_hash_set_changes": set_change_summary("evidence_string_hash_sets"),
            "canary_attack_success_pairs": sum(
                bool(row["canary"]["attack_success_under_frozen_definition"]) for row in pair_rows
            ),
            "canary_present_in_raw_pairs": sum(
                bool(row["canary"]["present_in_injected_raw_text"]) for row in pair_rows
            ),
            "canary_present_in_prediction_pairs": sum(
                bool(row["canary"]["present_in_injected_prediction"]) for row in pair_rows
            ),
            "interpretation": (
                "These measures quantify output displacement under the bundled injected-source condition. "
                "They do not identify unsupported facts or factual corruption without task-matched gold labels."
            ),
        },
        "sample_characteristics": {
            "context_count": len(context_rows),
            "groups": dict(sorted(groups.items())),
            "unique_report_count": len(reports),
            "reports": dict(sorted(reports.items())),
            "context_characters": {
                "min": min(sample_chars),
                "median": quantile(sample_chars, 0.5),
                "max": max(sample_chars),
            },
            "context_words": {
                "min": min(sample_words),
                "median": quantile(sample_words, 0.5),
                "max": max(sample_words),
            },
            "schema_expressibility_proxy": {
                "status": "not_assessed",
                "reason": (
                    "No response-independent, validated mapping from CTI-HAL ATT&CK labels or short "
                    "contexts to the full MALOnt entity/relation schema is available. A post-hoc lexical "
                    "heuristic would confound schema expressibility with wording."
                ),
            },
        },
        "semantic_fixture_templates": dict(semantic_templates),
    }


def validate_analysis(
    source_audit: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    raw_responses: Sequence[Mapping[str, Any]],
    context_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    if source_audit.get("status") != "PASS":
        failures.append("frozen source hashes did not match the original run manifest")
    case_ids = [str(row.get("case_id")) for row in cases]
    prediction_ids = [str(row.get("case_id")) for row in predictions]
    raw_ids = [str(row.get("case_id")) for row in raw_responses]
    if len(cases) != 126 or len(set(case_ids)) != 126:
        failures.append("expected 126 unique frozen cases")
    if set(prediction_ids) != set(case_ids):
        failures.append("predictions are not a one-to-one completion of cases")
    if set(raw_ids) != set(case_ids):
        failures.append("raw responses are not a one-to-one completion of cases")
    if len(context_rows) != 14:
        failures.append("expected 14 context clusters")
    if any(
        row["conditions"][condition]["repeat_count"] != 3
        for row in context_rows
        for condition in CONDITIONS
    ):
        failures.append("a context-condition cluster does not contain three repeats")
    if len(pair_rows) != 42:
        failures.append("expected 42 clean/injected pairs")
    if metrics["injection_pair_analysis"]["comparable_semantic_pairs"] != 42:
        failures.append("not all clean/injected predictions are structurally comparable")
    verdict = "PASS_WITH_WARNINGS" if not failures else "FAIL_REPAIR"
    return {
        "gate_id": "G6_hosted_clustered_reanalysis_integrity",
        "stage": "offline_clustered_reanalysis",
        "hypothesis_or_mechanism": (
            "Frozen outcomes can be recomputed at the context-cluster level and clean/injected "
            "responses can be paired without issuing new hosted calls."
        ),
        "verdict": verdict,
        "input_artifacts": [record["path"] for record in source_audit.get("checked", [])],
        "command_or_check": "python src/reanalyze_hosted_clusters.py",
        "pass_criteria": {
            "source_integrity": "required frozen source artifacts match their original manifest hashes",
            "case_integrity": "126 unique cases with one prediction and raw response each",
            "cluster_integrity": "14 contexts x 3 conditions x 3 repeats",
            "pair_integrity": "42 aligned typed-evidence/injection pairs",
            "metric_semantics": "context is bootstrap unit; fixed order and bundled prompts disclosed",
            "network_calls": "none",
        },
        "metrics": {
            "cases": len(cases),
            "contexts": len(context_rows),
            "paired_responses": len(pair_rows),
            "source_hashes_checked": len(source_audit.get("checked", [])),
        },
        "failed_criteria": failures,
        "warnings": [
            "fixed non-counterbalanced condition order",
            "minimal-to-typed contrast bundles multiple prompt components",
            "14 deterministic sampled contexts do not define a population sampling frame",
            "CTI-HAL is not full MALOnt factual gold",
        ],
        "fail_interpretation": (
            "The stored run would not support context-clustered validity or paired output-displacement summaries."
        ),
        "next_allowed_step": (
            "bounded descriptive reporting with all warnings"
            if not failures
            else "repair reanalysis inputs or implementation"
        ),
    }


def claim_boundary_gate(g6: Mapping[str, Any]) -> dict[str, Any]:
    passing = g6.get("verdict") == "PASS_WITH_WARNINGS"
    return {
        "gate_id": "G8_hosted_clustered_claim_permission",
        "stage": "results_to_claims",
        "verdict": "PASS_WITH_WARNINGS" if passing else "FAIL_REPAIR",
        "input_artifacts": ["metrics.json", "context_outcomes.jsonl", "injection_pairs.jsonl"],
        "command_or_check": "deterministic claim-boundary rules embedded in reanalysis script",
        "pass_criteria": {
            "allowed": (
                "descriptive contract-validity fractions, context-cluster sensitivity intervals, "
                "and paired clean/injected response-set displacement"
            ),
            "prohibited": (
                "factual accuracy/corruption, robustness generalization, prompt-component causality, "
                "or latency effects"
            ),
        },
        "metrics": {"upstream_gate": g6.get("verdict")},
        "failed_criteria": [] if passing else ["upstream G6 did not pass"],
        "warnings": list(g6.get("warnings", [])),
        "fail_interpretation": "Claims would exceed the construct validity of the frozen experiment.",
        "next_allowed_step": "bounded manuscript wording" if passing else "repair G6",
    }


def render_readme(metrics: Mapping[str, Any], source_run: Path, semantic_run: Path) -> str:
    typed = metrics["conditions"]["typed_evidence"]
    injected = metrics["conditions"]["typed_evidence_injection"]
    minimal = metrics["conditions"]["minimal_json"]
    pair = metrics["injection_pair_analysis"]
    return f"""# Hosted Clustered Reanalysis

This is an offline reanalysis of the frozen 126-call hosted run. It issued **no
hosted/model/network calls** and did not repair or replace any stored response.

## Inputs

- Hosted source run: `{source_run.relative_to(PAPER_ROOT).as_posix()}`
- Semantic fixture run: `{semantic_run.relative_to(PAPER_ROOT).as_posix()}`
- Required hosted input hashes were checked against the frozen source manifest.

## Analysis contract

- Unit of clustering: 14 sampled CTI contexts (7 groups, 2 contexts/group).
- Nested observations: 3 repeats/context/condition.
- Valid fraction: calls with stored status exactly `valid`, divided by 3 in each context.
- Uncertainty: deterministic paired percentile bootstrap resampling whole contexts,
  10,000 replicates. The interval is a sensitivity summary over these observed
  contexts, not a population confidence interval.
- Semantic comparison: ID-insensitive normalized entity `(type, name)` sets and
  relation `(resolved subject, predicate, resolved object)` sets. Empty/empty
  Jaccard is defined as 1. Evidence text is represented by SHA-256-bearing owner
  records in the pair ledger.
- Schema expressibility is not scored: no response-independent validated mapping
  from these CTI-HAL contexts to full MALOnt gold exists.

## Exact descriptive results

- Valid calls: minimal JSON {minimal['valid_calls']}/42; typed evidence
  {typed['valid_calls']}/42; typed evidence + injection {injected['valid_calls']}/42.
- Clean typed vs injected pairs: {pair['aligned_pairs']} aligned,
  {pair['exact_prediction_equal_pairs']} exact-prediction equal,
  {pair['any_exact_prediction_displacement_pairs']} with any exact-prediction
  displacement, and {pair['semantic_output_changed_pairs']} with changed canonical
  semantic sets (including {pair['clean_nonempty_to_injected_empty_pairs']}
  clean-nonempty to injected-empty transitions).
- Mean combined-set Jaccard: {pair['mean_combined_semantic_set_jaccard']:.6f}.
- Canary success/presence: {pair['canary_attack_success_pairs']} frozen-definition
  successes; {pair['canary_present_in_raw_pairs']} raw responses and
  {pair['canary_present_in_prediction_pairs']} parsed predictions contained the canary.

## Design limitations

The minimal-to-typed contrast bundles ontology type restrictions, relation
signatures, evidence requirements, empty-output guidance, and untrusted-data
instructions. Conditions also ran in a fixed, non-counterbalanced order. The
paired differences therefore cannot isolate prompt components or temporal
effects. Output displacement under injection does not establish factual
corruption or unsupported facts because task-matched factual gold is absent.

## Reproduce

From the paper root:

```bash
python src/reanalyze_hosted_clusters.py \\
  --source-run {source_run.relative_to(PAPER_ROOT).as_posix()} \\
  --semantic-run {semantic_run.relative_to(PAPER_ROOT).as_posix()} \\
  --output reproduced_runs/hosted_clustered_reanalysis_20260809
```

The command refuses to overwrite a non-empty output directory unless
`--overwrite` is given.
"""


def run(
    *,
    source_run: Path,
    semantic_run: Path,
    output: Path,
    seed: int = DEFAULT_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    overwrite: bool = False,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    source_run = source_run.resolve()
    semantic_run = semantic_run.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    required = [
        source_run / "cases.jsonl",
        source_run / "predictions.jsonl",
        source_run / "raw_responses.jsonl",
    ]
    source_audit = verify_frozen_source(source_run, required)
    if source_audit["status"] != "PASS":
        raise ReanalysisContractError(f"frozen source integrity failed: {source_audit['failures']}")
    cases = read_jsonl(required[0])
    predictions = read_jsonl(required[1])
    raw_responses = read_jsonl(required[2])
    semantic_cases_path = semantic_run / "cases.jsonl"
    semantic_cases = read_jsonl(semantic_cases_path)
    context_rows = build_context_outcomes(cases, predictions)
    pair_rows = build_injection_pairs(cases, predictions, raw_responses)
    template_inventory = semantic_template_inventory(semantic_cases)
    metrics = compute_metrics(
        context_rows,
        pair_rows,
        template_inventory,
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    g6 = validate_analysis(
        source_audit,
        cases,
        predictions,
        raw_responses,
        context_rows,
        pair_rows,
        metrics,
    )
    g8 = claim_boundary_gate(g6)

    context_path = output / "context_outcomes.jsonl"
    pair_path = output / "injection_pairs.jsonl"
    sample_path = output / "sample_characteristics.jsonl"
    metrics_path = output / "metrics.json"
    source_audit_path = output / "source_integrity.json"
    g6_path = output / "gate_G6_clustered_reanalysis.json"
    g8_path = output / "gate_G8_claim_permission.json"
    readme_path = output / "README.md"
    write_jsonl(context_path, context_rows)
    write_jsonl(pair_path, pair_rows)
    write_jsonl(
        sample_path,
        [
            {key: value for key, value in row.items() if key != "conditions"}
            for row in context_rows
        ],
    )
    write_json(metrics_path, metrics)
    write_json(source_audit_path, source_audit)
    write_json(g6_path, g6)
    write_json(g8_path, g8)
    readme_path.write_text(render_readme(metrics, source_run, semantic_run), encoding="utf-8")

    finished_at = utc_now()
    outputs = [
        context_path,
        pair_path,
        sample_path,
        metrics_path,
        source_audit_path,
        g6_path,
        g8_path,
        readme_path,
    ]
    manifest = build_manifest(
        run_id=output.name,
        command=list(command or ["python", "src/reanalyze_hosted_clusters.py"]),
        seed=seed,
        started_at=started_at,
        finished_at=finished_at,
        inputs=[*required, source_run / "manifest.json", semantic_cases_path, Path(__file__)],
        outputs=outputs,
        base=PAPER_ROOT,
        dependencies=(),
        extra={
            "execution_mode": "offline_only_no_hosted_calls",
            "bootstrap_replicates": bootstrap_replicates,
            "source_run": source_run.relative_to(PAPER_ROOT).as_posix(),
            "semantic_run": semantic_run.relative_to(PAPER_ROOT).as_posix(),
        },
    )
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    manifest_audit = verify_output_manifest(manifest_path)
    write_json(output / "manifest_verification.json", manifest_audit)
    if manifest_audit["status"] != "PASS":
        raise ReanalysisContractError(
            f"output manifest verification failed: {manifest_audit['failures']}"
        )
    return {"metrics": metrics, "g6": g6, "g8": g8, "output": str(output)}


def _paper_relative(path: Path) -> Path:
    return path if path.is_absolute() else PAPER_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--semantic-run", type=Path, default=DEFAULT_SEMANTIC_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = ["python", "src/reanalyze_hosted_clusters.py", *(argv or [])]
    result = run(
        source_run=_paper_relative(args.source_run),
        semantic_run=_paper_relative(args.semantic_run),
        output=_paper_relative(args.output),
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        overwrite=args.overwrite,
        command=command,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["g6"]["verdict"] == "PASS_WITH_WARNINGS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

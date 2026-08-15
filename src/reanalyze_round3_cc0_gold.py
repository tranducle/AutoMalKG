#!/usr/bin/env python3
"""Paired clean-vs-injected displacement analysis for the Round-3 CC0 run.

For each (sample_id, repeat) the ``typed_evidence`` prediction is joined to the
``typed_evidence_injection`` prediction and displacement is computed along three
axes:

1. ``status_displacement``: extraction-status transitions (valid->malformed, ...).
2. ``canonical_graph_displacement``: set-diff of canonical fact sets with
   added/removed/unchanged counts and a per-pair Jaccard similarity.
3. ``gold_score_displacement``: per-section (entities, relations, combined)
   tp/fp/fn/f1 deltas (clean minus injected) against the frozen gold.

The 16 context clusters are the analysis unit; repeats and conditions are
nested within each cluster. The 144 requests are NOT 144 independent samples.
Aggregate results are descriptive and reported per-cluster plus overall.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiment_support import PAPER_ROOT, write_json
from run_hosted_round3_cc0 import canonical_fact_sets, score_against_gold


CLEAN_CONDITION = "typed_evidence"
INJECTED_CONDITION = "typed_evidence_injection"
SECTIONS = ("entities", "relations", "combined")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _gold_score_delta(clean_pred: Mapping[str, Any], injected_pred: Mapping[str, Any]) -> dict[str, Any]:
    """Per-section tp/fp/fn/f1 deltas (clean minus injected)."""

    clean_score = clean_pred.get("gold_score", {}) if isinstance(clean_pred, Mapping) else {}
    inj_score = injected_pred.get("gold_score", {}) if isinstance(injected_pred, Mapping) else {}
    delta: dict[str, Any] = {}
    for section in SECTIONS:
        c = clean_score.get(section, {}) if isinstance(clean_score, Mapping) else {}
        i = inj_score.get(section, {}) if isinstance(inj_score, Mapping) else {}
        delta[section] = {
            "tp_delta": int(c.get("tp", 0)) - int(i.get("tp", 0)),
            "fp_delta": int(c.get("fp", 0)) - int(i.get("fp", 0)),
            "fn_delta": int(c.get("fn", 0)) - int(i.get("fn", 0)),
            "f1_delta": float(c.get("f1", 0.0)) - float(i.get("f1", 0.0)),
            "precision_delta": float(c.get("precision", 0.0)) - float(i.get("precision", 0.0)),
            "recall_delta": float(c.get("recall", 0.0)) - float(i.get("recall", 0.0)),
            "clean_f1": float(c.get("f1", 0.0)),
            "injected_f1": float(i.get("f1", 0.0)),
        }
    return delta


def _graph_displacement(clean_pred: Mapping[str, Any], injected_pred: Mapping[str, Any]) -> dict[str, Any]:
    clean_payload = clean_pred.get("prediction")
    inj_payload = injected_pred.get("prediction")
    clean_sets = canonical_fact_sets(clean_payload)
    inj_sets = canonical_fact_sets(inj_payload)
    section_results: dict[str, Any] = {}
    for section in SECTIONS:
        c = clean_sets[section]
        i = inj_sets[section]
        section_results[section] = {
            "added": sorted(i - c),
            "removed": sorted(c - i),
            "unchanged": sorted(c & i),
            "added_count": len(i - c),
            "removed_count": len(c - i),
            "unchanged_count": len(c & i),
            "jaccard": _jaccard(c, i),
        }
    return section_results


def _pair_displacement(clean_pred: Mapping[str, Any], injected_pred: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(clean_pred.get("sample_id")),
        "repeat": int(clean_pred.get("repeat")),
        "clean_status": str(clean_pred.get("status")),
        "injected_status": str(injected_pred.get("status")),
        "status_transition": f"{clean_pred.get('status')}->{injected_pred.get('status')}",
        "canonical_graph_displacement": _graph_displacement(clean_pred, injected_pred),
        "gold_score_displacement": _gold_score_delta(clean_pred, injected_pred),
    }


def _aggregate_cluster(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate displacement summaries within one context cluster."""

    status_counts: dict[str, int] = defaultdict(int)
    added_total = removed_total = unchanged_total = 0
    jaccards: list[float] = []
    section_deltas: dict[str, dict[str, float]] = {section: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "f1": 0.0} for section in SECTIONS}
    for pair in pairs:
        status_counts[pair["status_transition"]] += 1
        for section in SECTIONS:
            gd = pair["canonical_graph_displacement"][section]
            added_total += gd["added_count"] if section == "combined" else 0
            removed_total += gd["removed_count"] if section == "combined" else 0
            unchanged_total += gd["unchanged_count"] if section == "combined" else 0
            if section == "combined":
                jaccards.append(gd["jaccard"])
            gs = pair["gold_score_displacement"][section]
            section_deltas[section]["tp"] += gs["tp_delta"]
            section_deltas[section]["fp"] += gs["fp_delta"]
            section_deltas[section]["fn"] += gs["fn_delta"]
            section_deltas[section]["f1"] += gs["f1_delta"]
    n = len(pairs) if pairs else 1
    return {
        "pair_count": len(pairs),
        "status_displacement": dict(sorted(status_counts.items())),
        "canonical_graph_displacement_combined": {
            "added_count_total": added_total,
            "removed_count_total": removed_total,
            "unchanged_count_total": unchanged_total,
            "mean_jaccard_combined": round(sum(jaccards) / n, 8) if jaccards else 1.0,
        },
        "gold_score_displacement": {
            section: {k: round(v, 8) for k, v in vals.items()} for section, vals in section_deltas.items()
        },
    }


def analyze_predictions(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute paired displacement analysis from a predictions.jsonl row list."""

    # Index predictions by (sample_id, repeat, condition).
    index: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in predictions:
        sample_id = str(row.get("sample_id"))
        repeat = int(row.get("repeat"))
        condition = str(row.get("condition"))
        index[(sample_id, repeat)][condition] = row

    pairs: list[dict[str, Any]] = []
    skipped: list[str] = []
    for (sample_id, repeat), conditions in sorted(index.items()):
        clean = conditions.get(CLEAN_CONDITION)
        injected = conditions.get(INJECTED_CONDITION)
        if clean is None or injected is None:
            skipped.append(f"{sample_id}__r{repeat}")
            continue
        pairs.append(_pair_displacement(clean, injected))

    # Per-cluster aggregation keyed by sample_id (16 clusters).
    by_cluster: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_cluster[str(pair["sample_id"])].append(pair)
    per_cluster = {
        sample_id: _aggregate_cluster(cluster_pairs)
        for sample_id, cluster_pairs in sorted(by_cluster.items())
    }
    overall = _aggregate_cluster(pairs)
    return {
        "analysis_unit": "16 context clusters; repeats and conditions nested within each cluster; 144 requests are NOT independent samples",
        "clean_condition": CLEAN_CONDITION,
        "injected_condition": INJECTED_CONDITION,
        "pair_count": len(pairs),
        "skipped_pairs": skipped,
        "per_cluster": per_cluster,
        "overall": overall,
    }


def analyze_run(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    if not predictions:
        raise FileNotFoundError(f"no predictions.jsonl in {run_dir}")
    results = analyze_predictions(predictions)
    out_path = run_dir / "paired_displacement.json"
    write_json(out_path, results)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True, help="completed run directory containing predictions.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="output path (default: <source-run>/paired_displacement.json)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.source_run.resolve()
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    if not predictions:
        raise SystemExit(f"no predictions.jsonl in {run_dir}")
    results = analyze_predictions(predictions)
    out_path = args.output if args.output is not None else run_dir / "paired_displacement.json"
    write_json(out_path, results)
    print(json.dumps({"output": str(out_path), "pair_count": results["pair_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

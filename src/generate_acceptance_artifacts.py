"""Generate manuscript tables and figures only from completed run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.tools.academic_style import apply_style, figsize, find_overlaps, save_fig


CONDITION_LABELS = {
    "minimal_json": "Minimal JSON",
    "typed_evidence": "Typed + evidence",
    "typed_evidence_injection": "Typed + evidence + injection",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected an object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"expected an object at {path}:{number}")
        rows.append(value)
    return rows


def artifact_path(path: Path) -> str:
    """Return a paper-root-relative path without leaking a local workstation path."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PAPER_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def require_completed_hosted_run(run_dir: Path) -> None:
    config = read_json(run_dir / "run_config.json")
    if config.get("state") != "completed":
        raise RuntimeError(f"hosted run is not completed: {run_dir}")
    for name in ("manifest.json", "metrics.json", "predictions.jsonl"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(run_dir / name)


def fmt_rate(value: Any) -> str:
    if value is None:
        return "--"
    return f"{100.0 * float(value):.1f}"


def fmt_ms(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.1f}"


def assert_no_text_overlaps(fig: Any, name: str) -> None:
    overlaps = find_overlaps(fig)
    if overlaps:
        rendered = "; ".join(str(item) for item in overlaps[:10])
        raise RuntimeError(f"text overlap in {name}: {rendered}")


def make_architecture_figure(output_dir: Path) -> list[Path]:
    """Render the implemented fail-closed state flow without efficacy claims."""

    apply_style("double")
    fig, ax = plt.subplots(figsize=figsize("double", ratio=0.53))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        label: str,
        *,
        face: str = "#E6F2FA",
        edge: str = "#0072B2",
        fontsize: float = 7.4,
    ) -> tuple[float, float]:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            zorder=3,
        )
        return (x + width / 2, y + height / 2)

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = "#4B5563",
        linestyle: str = "solid",
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.0,
                color=color,
                linestyle=linestyle,
                shrinkA=1,
                shrinkB=1,
                zorder=1,
            )
        )

    box(0.02, 0.67, 0.15, 0.17, "Untrusted CTI\ntext chunks", face="#F2F4F7", edge="#6B7280")
    box(0.22, 0.67, 0.15, 0.17, "Hosted LLM\nparser")
    box(0.42, 0.67, 0.17, 0.17, "Parse + typed\nschema validation")
    box(0.64, 0.67, 0.17, 0.17, "Canonical merge\n+ provenance")
    box(0.83, 0.67, 0.15, 0.17, "Temporary OWL world\n+ Pellet")
    arrow((0.17, 0.755), (0.22, 0.755))
    arrow((0.37, 0.755), (0.42, 0.755))
    arrow((0.59, 0.755), (0.64, 0.755))
    arrow((0.81, 0.755), (0.83, 0.755))

    box(0.07, 0.25, 0.20, 0.17, "STOP: extraction failure\n(parse/schema/provider)", face="#FCE8E6", edge="#D55E00")
    box(0.35, 0.25, 0.18, 0.17, "STOP: INCONSISTENT\nor UNVERIFIED", face="#FCE8E6", edge="#D55E00")
    box(0.60, 0.25, 0.17, 0.17, "PASS: CONSISTENT\napply Rule 1", face="#E7F5EF", edge="#009E73")
    box(0.83, 0.25, 0.15, 0.17, "PASS: graph +\nproof ledger", face="#E7F5EF", edge="#009E73")
    arrow((0.505, 0.67), (0.17, 0.42), color="#D55E00", linestyle="dashed")
    arrow((0.905, 0.67), (0.44, 0.42), color="#D55E00", linestyle="dashed")
    arrow((0.905, 0.67), (0.685, 0.42), color="#009E73")
    arrow((0.77, 0.335), (0.83, 0.335), color="#009E73")

    label_box = {"facecolor": "white", "alpha": 0.92, "edgecolor": "none", "pad": 1.2}
    ax.text(0.305, 0.52, "typed failure; no graph", ha="center", va="center", fontsize=7.4, color="#9A3412", bbox=label_box)
    ax.text(0.655, 0.52, "fail closed", ha="center", va="center", fontsize=7.4, color="#9A3412", bbox=label_box)
    ax.text(0.86, 0.52, "verified status only", ha="center", va="center", fontsize=7.4, color="#006B4F", bbox=label_box)
    ax.text(
        0.5,
        0.07,
        "Structural consistency is not factual verification; well-typed false claims can pass.",
        ha="center",
        va="center",
        fontsize=7.4,
        fontstyle="italic",
        color="#374151",
    )
    fig.tight_layout(pad=0.2)
    assert_no_text_overlaps(fig, "architecture_contract")
    return save_fig(fig, output_dir / "architecture_contract", dpi=600)


def make_semantic_scaling_figure(
    predictions: Sequence[dict[str, Any]], output_dir: Path
) -> list[Path]:
    by_size: dict[int, list[float]] = defaultdict(list)
    for row in predictions:
        if row.get("family") != "scale" or row.get("warmup"):
            continue
        by_size[int(row["graph_size"])].append(float(row["runtime_ms"]))
    sizes = sorted(by_size)
    if not sizes or any(len(by_size[size]) < 2 for size in sizes):
        raise RuntimeError("semantic scaling predictions are incomplete")

    apply_style("single")
    fig, ax = plt.subplots(figsize=figsize("single", ratio=0.76))
    box = ax.boxplot(
        [by_size[size] for size in sizes],
        tick_labels=[f"{size:,}" for size in sizes],
        patch_artist=True,
        widths=0.58,
        showfliers=True,
        medianprops={"color": "#D55E00", "linewidth": 1.5},
        whiskerprops={"color": "#4B5563"},
        capprops={"color": "#4B5563"},
        boxprops={"edgecolor": "#0072B2", "linewidth": 1.0},
        flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": "#0072B2"},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#56B4E9")
        patch.set_alpha(0.42)
    ax.set_xlabel("Licensed chains (two asserted relations per chain)")
    ax.set_ylabel("Pellet runtime (ms)")
    ax.set_title("Local OWL/SWRL runtime distribution (10 repetitions per size)")
    ax.set_axisbelow(True)
    fig.tight_layout()
    assert_no_text_overlaps(fig, "semantic_scaling")
    return save_fig(fig, output_dir / "semantic_scaling", dpi=600)


def make_hosted_contract_figure(metrics: dict[str, Any], output_dir: Path) -> list[Path]:
    conditions = list(CONDITION_LABELS)
    rows = [metrics["conditions"][condition] for condition in conditions]
    valid = [100.0 * float(row["valid_rate"]) for row in rows]
    prediction_stability = [
        100.0 * float(row["exact_prediction_repeat_stability_rate"]) for row in rows
    ]

    apply_style("single")
    fig, ax = plt.subplots(figsize=figsize("single", ratio=0.80))
    x = list(range(len(conditions)))
    width = 0.34
    ax.bar(
        [value - width / 2 for value in x],
        valid,
        width,
        label="Valid responses",
        color="#0072B2",
    )
    ax.bar(
        [value + width / 2 for value in x],
        prediction_stability,
        width,
        label="Canonical-response stability (14 groups)",
        color="#E69F00",
    )
    ax.set_xticks(x, ["Minimal\nJSON", "Typed +\nevidence", "Typed + evidence\n+ injection"])
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Hosted response-contract stress")
    for index, row in enumerate(rows):
        stable_count = int(round(14 * float(row["exact_prediction_repeat_stability_rate"])))
        ax.text(
            index + width / 2,
            prediction_stability[index] + 2.0,
            f"{stable_count}/14",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1)
    ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=0.32, left=0.18, right=0.98, top=0.88)
    assert_no_text_overlaps(fig, "hosted_contract")
    return save_fig(fig, output_dir / "hosted_contract", dpi=600)


def write_semantic_table(metrics: dict[str, Any], target: Path) -> None:
    consistency = metrics["consistency"]
    swrl = metrics["swrl_controls"]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Controlled task} & \textbf{$n$} & \textbf{TP} & \textbf{TN} & \textbf{Primary result} \\",
        r"\midrule",
        (
            "OWL consistency & "
            f"{consistency['attempted']} & "
            f"{consistency['confusion_matrix_inconsistent_is_positive']['tp']} & "
            f"{consistency['confusion_matrix_inconsistent_is_positive']['tn']} & "
            f"TPR/TNR={consistency['tpr']:.2f}/{consistency['tnr']:.2f} \\\\"
        ),
        (
            "SWRL edge completion & "
            f"{swrl['attempted']} & "
            f"{swrl['edge_confusion_matrix']['tp']} & "
            f"{swrl['edge_confusion_matrix']['tn']} & "
            f"P/R/F1={swrl['edge_precision']:.2f}/{swrl['edge_recall']:.2f}/{swrl['edge_f1']:.2f} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")


def write_hosted_table(metrics: dict[str, Any], target: Path) -> None:
    lines = [
        r"\begin{tabular}{l@{\hspace{0.8em}}rrrrr}",
        r"\toprule",
        r"\textbf{Prompt condition} & \textbf{Calls} & \textbf{Valid (\%)} & \textbf{Stable (\%)} & \textbf{Median (ms)} & \textbf{p95 (ms)} \\",
        r"\midrule",
    ]
    for condition, label in CONDITION_LABELS.items():
        row = metrics["conditions"][condition]
        latency = row["latency_seconds"]
        lines.append(
            f"{label} & {row['completed_calls']} & {fmt_rate(row['valid_rate'])} & "
            f"{fmt_rate(row['exact_prediction_repeat_stability_rate'])} & "
            f"{fmt_ms(latency['median'] * 1000 if latency['median'] is not None else None)} & "
            f"{fmt_ms(latency['p95'] * 1000 if latency['p95'] is not None else None)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    target.write_text("\n".join(lines), encoding="utf-8")


def write_alternative_table(metrics: dict[str, Any], target: Path) -> None:
    """Write the independently specified violation-fixture stage comparison."""

    rows = metrics["summaries"]["synthetic_fixture"]["methods"]
    labels = {
        "json_parse_only": "JSON parse only",
        "strict_json_schema": "Strict JSON Schema",
        "production_canonicalization": "Production contract + canonicalization",
        "full_automalkg": "Full AutoMalKG",
    }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Assurance stage} & \textbf{Accepted} & \textbf{Rejected} & \textbf{Silent violations} & \textbf{Provenance} \\",
        r"\midrule",
    ]
    for key, label in labels.items():
        row = rows[key]
        provenance = row["provenance_completeness"]
        provenance_text = f"{100.0 * provenance:.0f}\\%" if provenance is not None else "--"
        lines.append(
            f"{label} & {row['accepted']} & {row['rejected']} & "
            f"{row['silent_acceptance_of_known_violations']}/{row['known_violation_cases']} & {provenance_text} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    target.write_text("\n".join(lines), encoding="utf-8")


def write_e2e_attrition_table(metrics: dict[str, Any], target: Path) -> None:
    """Write the frozen-response production-pipeline stage ledger summary."""

    counts = metrics["stage_counts"]
    rows = [
        ("Raw-response hash verified", counts["raw_hash_verified"]),
        ("JSON parsed", counts["json_parse_passed"]),
        ("Schema-valid", counts["schema_passed"]),
        ("Non-empty canonical graph", counts["nonempty_canonical_graph"]),
        ("Ontology load validated", counts["ontology_load_validated"]),
        ("Pellet consistent", counts["pellet_consistent"]),
        ("Verified artifact serialized", counts["serialized_verified_artifacts"]),
    ]
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"\textbf{Frozen-response stage} & \textbf{Cases passing} \\",
        r"\midrule",
    ]
    total = metrics.get("case_count", counts["raw_hash_verified"])
    lines.extend(f"{label} & {count}/{total} \\\\" for label, count in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    target.write_text("\n".join(lines), encoding="utf-8")


def generate(
    semantic_run: Path,
    hosted_run: Path,
    e2e_run: Path,
    alternative_run: Path,
    clustered_run: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require_completed_hosted_run(hosted_run)
    semantic_metrics = read_json(semantic_run / "metrics.json")
    hosted_metrics = read_json(hosted_run / "metrics.json")
    e2e_metrics = read_json(e2e_run / "metrics.json")
    alternative_metrics = read_json(alternative_run / "metrics.json")
    clustered_metrics = read_json(clustered_run / "metrics.json")
    semantic_predictions = read_jsonl(semantic_run / "predictions.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    written.extend(make_architecture_figure(output_dir))
    written.extend(make_semantic_scaling_figure(semantic_predictions, output_dir))
    written.extend(make_hosted_contract_figure(hosted_metrics, output_dir))
    semantic_table = output_dir / "semantic_results_table.tex"
    hosted_table = output_dir / "hosted_results_table.tex"
    alternative_table = output_dir / "alternative_ablation_table.tex"
    e2e_table = output_dir / "hosted_e2e_attrition_table.tex"
    write_semantic_table(semantic_metrics, semantic_table)
    write_hosted_table(hosted_metrics, hosted_table)
    write_alternative_table(alternative_metrics, alternative_table)
    write_e2e_attrition_table(e2e_metrics, e2e_table)
    written.extend([semantic_table, hosted_table, alternative_table, e2e_table])

    summary = {
        "semantic_run": artifact_path(semantic_run),
        "hosted_run": artifact_path(hosted_run),
        "e2e_run": artifact_path(e2e_run),
        "alternative_run": artifact_path(alternative_run),
        "clustered_run": artifact_path(clustered_run),
        "inputs": {
            "semantic_metrics": artifact_path(semantic_run / "metrics.json"),
            "semantic_predictions": artifact_path(semantic_run / "predictions.jsonl"),
            "hosted_metrics": artifact_path(hosted_run / "metrics.json"),
            "e2e_metrics": artifact_path(e2e_run / "metrics.json"),
            "alternative_metrics": artifact_path(alternative_run / "metrics.json"),
            "clustered_metrics": artifact_path(clustered_run / "metrics.json"),
        },
        "outputs": [artifact_path(path) for path in written],
        "figure_policy": "observation-level data only; shared academic_style; 600 dpi PNG plus vector PDF",
    }
    summary_path = output_dir / "artifact_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["outputs"].append(artifact_path(summary_path))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-run", type=Path, required=True)
    parser.add_argument("--hosted-run", type=Path, required=True)
    parser.add_argument("--e2e-run", type=Path, required=True)
    parser.add_argument("--alternative-run", type=Path, required=True)
    parser.add_argument("--clustered-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        json.dumps(
            generate(
                args.semantic_run,
                args.hosted_run,
                args.e2e_run,
                args.alternative_run,
                args.clustered_run,
                args.output_dir,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Check that a manuscript still agrees with the run artefacts it reports.

Two independent checks, both read-only:

1. **Exhibit sync** - regenerate the acceptance-loop table fragments from the runs named on
   the command line and compare them against the fragments shipped with the manuscript.
   Differences are reported unless the shipped fragment's own comment header declares the
   difference as a post-generation edit in the form ``"old" -> "new"``.
2. **Prose sync** - recompute every reported quantity that is mechanically derivable from
   ``metrics.json`` and ``paired_displacement.json`` of the hosted run, format it the way the
   manuscript formats it, and require the resulting string to appear in the manuscript source.

A re-run of the experiments changes the artefacts. Anything the manuscript states in prose is
hand-written and no generator touches it, so without this check a re-run silently regenerates
the tables and leaves the prose stale. Failure here means the manuscript and the artefacts
disagree; it does not say which of the two is wrong.

Exit codes: 0 = in sync, 1 = drift found, 2 = an input is unusable.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

GENERATOR = "generate_acceptance_artifacts.py"
FRAGMENTS = (
    "alternative_ablation_table.tex",
    "hosted_results_table.tex",
    "hosted_e2e_attrition_table.tex",
)
# "old" -> "new" recorded in a fragment's comment header as a deliberate post-generation edit.
EDIT_DECLARATION = re.compile(r'"([^"]+)"\s*->\s*(?:\n%\s*)?"([^"]+)"')


@dataclass(frozen=True)
class Assertion:
    """One reported quantity.

    ``pattern`` anchors the quantity in its own wording and captures the value(s) the
    manuscript states; ``expected`` is what the artefact says they must be. Checking a
    captured value rather than the mere presence of a string matters: a bare number such as
    ``0.903`` occurs at more than one site, so a presence test passes while one of those sites
    is stale. Every match must agree, which is what makes a second, unupdated site fail.
    """

    label: str
    source: str
    pattern: str
    expected: Tuple[str, ...]
    # Number of sites the quantity is known to occupy. A pattern that matches some sites but
    # not all of them is the one failure mode a match-and-compare check cannot see by itself:
    # it reports agreement having silently never looked at the stale site.
    sites: int = 1


@dataclass(frozen=True)
class Finding:
    check: str
    label: str
    detail: str


def _load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise SystemExit(f"required artefact not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"artefact is not valid JSON: {path}: {exc}")


def _strip_comments(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not line.lstrip().startswith("%"))


def _declared_edits(text: str) -> List[Tuple[str, str]]:
    header = "\n".join(line for line in text.split("\n") if line.lstrip().startswith("%"))
    return EDIT_DECLARATION.findall(header)


def check_exhibits(
    project: Path, generator: Path, runs: Dict[str, Path], figures_dir: Path
) -> List[Finding]:
    """Regenerate the fragments and diff them against what the manuscript ships."""
    findings: List[Finding] = []
    scratch = Path(tempfile.mkdtemp(prefix="manuscript_sync_"))
    try:
        cmd = [sys.executable, str(generator)]
        for name, path in runs.items():
            cmd += [f"--{name}-run", str(path)]
        cmd += ["--output-dir", str(scratch)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project))
        if proc.returncode != 0:
            return [Finding("exhibit", GENERATOR, f"generator failed: {proc.stderr.strip()[-400:]}")]

        for name in FRAGMENTS:
            shipped_path = figures_dir / name
            fresh_path = scratch / name
            if not shipped_path.exists():
                findings.append(Finding("exhibit", name, "shipped fragment is missing"))
                continue
            if not fresh_path.exists():
                findings.append(Finding("exhibit", name, "generator produced no such fragment"))
                continue
            shipped_text = shipped_path.read_text(encoding="utf-8")
            fresh = _strip_comments(fresh_path.read_text(encoding="utf-8"))
            for old, new in _declared_edits(shipped_text):
                fresh = fresh.replace(old, new)
            if _strip_comments(shipped_text).strip() != fresh.strip():
                findings.append(
                    Finding(
                        "exhibit",
                        name,
                        "regenerated fragment differs from the shipped one beyond the edits its "
                        "header declares; the shipped table is stale or was hand-edited silently",
                    )
                )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return findings


def build_assertions(metrics: dict, paired: dict) -> List[Assertion]:
    """Derive every prose quantity the hosted-run artefacts determine."""
    conditions = metrics["conditions"]
    factual = metrics["factual_set_metrics"]
    status = metrics["status_distribution"]
    overall = paired["overall"]
    gold = overall["gold_score_displacement"]["combined"]
    clusters = paired["per_cluster"]
    groups = conditions["typed_evidence"]["repeat_groups"]

    order = ("minimal_json", "typed_evidence", "typed_evidence_injection")
    valid = [int(round(conditions[c]["valid_rate"] * conditions[c]["completed_calls"])) for c in order]
    calls = [conditions[c]["completed_calls"] for c in order]
    canonical = [int(round(conditions[c]["exact_prediction_repeat_stability_rate"] * groups)) for c in order]
    raw = [int(round(conditions[c]["exact_raw_repeat_stability_rate"] * groups)) for c in order]
    latency = [conditions[c]["latency_seconds"]["median"] for c in order]

    clean_f1 = factual["typed_evidence"]["combined"]["micro_f1"]
    inj_f1 = factual["typed_evidence_injection"]["combined"]["micro_f1"]

    # Context-unit displacement: each cluster's summed F1 delta averaged over its own pairs.
    per_context = [
        c["gold_score_displacement"]["combined"]["f1"] / c["pair_count"] * 100.0
        for c in clusters.values()
    ]
    zero_contexts = sum(1 for v in per_context if abs(v) < 1e-9)

    def signed(value: float, digits: int = 0) -> str:
        """Signed number the way the manuscript sets it: the whole token in math mode."""
        return f"{value:+.{digits}f}"

    typed = factual["typed_evidence"]["combined"]
    num = r"([0-9]+(?:\.[0-9]+)?)"
    sgn = r"\$([+-][0-9]+(?:\.[0-9]+)?)\$"
    frac = r"([0-9]+/[0-9]+)"

    return [
        Assertion("completed calls", "metrics.case_count_completed",
                  rf"All {num} requests completed",
                  (str(metrics["case_count_completed"]),)),
        Assertion("valid / invalid split", "metrics.status_distribution",
                  rf"{num} responses were valid and {num} schema-invalid",
                  (str(status["valid"]), str(status["schema_invalid"]))),
        Assertion("per-condition validity (results)", "metrics.conditions.*.valid_rate",
                  # The three fractions no longer sit in one clause: the minimal condition was
                  # relabelled a floor control and now carries its own justification between
                  # itself and the other two. ``.*?`` spans that, and only that, because the
                  # paragraph is a single physical line and ``.`` does not cross newlines.
                  rf"minimal floor control produced {frac} valid records.*?"
                  rf"The typed-evidence prompt produced {frac}[^;]*; the injection condition {frac}",
                  tuple(f"{v}/{c}" for v, c in zip(valid, calls))),
        Assertion("per-condition validity (RQ5)", "metrics.conditions.*.valid_rate",
                  rf"validity was {frac}, {frac}, and {frac} across conditions",
                  tuple(f"{v}/{c}" for v, c in zip(valid, calls))),
        Assertion("pooled micro-F1 clean vs injected", "metrics.factual_set_metrics.*.combined.micro_f1",
                  rf"micro-F1 fell from {num} \(clean\) to {num} \(injected\)",
                  (f"{clean_f1:.3f}", f"{inj_f1:.3f}")),
        Assertion("pooled F1 difference", "derived from the two pooled micro-F1 values",
                  rf"a difference of {num} percentage points",
                  (f"{(clean_f1 - inj_f1) * 100:.2f}",)),
        Assertion("pooled count deltas", "paired.overall.gold_score_displacement.combined",
                  rf"pooled count deltas of {sgn} true positives, {sgn} false positives, "
                  rf"and {sgn} false negatives",
                  (signed(gold["tp"]), signed(gold["fp"]), signed(gold["fn"]))),
        Assertion("context-unit mean displacement", "paired.per_cluster",
                  rf"mean paired displacement was {sgn}\\,pp",
                  (signed(statistics.mean(per_context), 1),)),
        Assertion("context-unit SD", "paired.per_cluster",
                  rf"standard deviation, SD, {num}\\,pp",
                  (f"{statistics.stdev(per_context):.1f}",)),
        Assertion("displacement range", "paired.per_cluster",
                  rf"range {sgn} to {sgn}\\,pp",
                  (signed(min(per_context), 1), signed(max(per_context), 1))),
        Assertion("contexts with no displacement", "paired.per_cluster",
                  rf"{num} of the {num} contexts showed no displacement",
                  (str(zero_contexts), str(len(clusters)))),
        Assertion("contexts carrying the displacement (RQ5)", "paired.per_cluster",
                  rf"displacement concentrated in {num} of the {num} contexts",
                  (str(len(clusters) - zero_contexts), str(len(clusters)))),
        Assertion("matched pair count", "paired.pair_count",
                  rf"analysis over {num} matched pairs", (str(paired["pair_count"]),)),
        Assertion("status transitions", "paired.overall.status_displacement",
                  rf"found {frac} valid-to-valid status transitions",
                  (f"{overall['status_displacement']['valid->valid']}/{paired['pair_count']}",)),
        Assertion("mean Jaccard (every site)", "paired.overall.canonical_graph_displacement_combined",
                  rf"mean Jaccard[^0-9]{{0,200}}{num}",
                  (f"{overall['canonical_graph_displacement_combined']['mean_jaccard_combined']:.3f}",),
                  sites=3),  # results, RQ5 answer, conclusion
        Assertion("canonical repeat stability", "metrics.exact_prediction_repeat_stability_rate",
                  rf"it held for {frac}, {frac}, and {frac} groups",
                  tuple(f"{v}/{groups}" for v in canonical)),
        Assertion("raw repeat stability", "metrics.exact_raw_repeat_stability_rate",
                  rf"byte-identical raw response text it held for {frac}, {frac}, and {frac}",
                  tuple(f"{v}/{groups}" for v in raw)),
        Assertion("median latencies", "metrics.conditions.*.latency_seconds.median",
                  rf"Median latency was {num}\\,s \(minimal\), {num}\\,s \(typed\), "
                  rf"and {num}\\,s \(injection\)",
                  tuple(f"{v:.2f}" for v in latency)),
        Assertion("run cost", "metrics.cost_accounting.known_cost_usd",
                  rf"run cost US\\\${num}",
                  (f"{metrics['cost_accounting']['known_cost_usd']:.4f}",)),
        Assertion("typed pooled decomposition", "metrics.factual_set_metrics.typed_evidence.combined",
                  rf"micro-F1 of {num} decomposes as precision {num} and recall {num}",
                  (f"{typed['micro_f1']:.3f}", f"{typed['micro_precision']:.3f}",
                   f"{typed['micro_recall']:.3f}")),
        Assertion("typed confusion counts", "metrics.factual_set_metrics.typed_evidence.combined",
                  rf"from {num} true positives, {num} false positives, and {num} false negatives",
                  (str(typed["tp"]), str(typed["fp"]), str(typed["fn"]))),
        Assertion("minimal pooled micro-F1", "metrics.factual_set_metrics.minimal_json.combined",
                  rf"the minimal condition reached {num}",
                  (f"{factual['minimal_json']['combined']['micro_f1']:.3f}",)),
    ]


def check_prose(manuscript: Path, assertions: Sequence[Assertion]) -> List[Finding]:
    try:
        text = manuscript.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read manuscript: {manuscript}: {exc}")
    findings: List[Finding] = []
    for item in assertions:
        matches = re.findall(item.pattern, text)
        if not matches:
            findings.append(
                Finding("prose", item.label,
                        f"the manuscript no longer states this quantity in the wording this "
                        f"check anchors on (pattern: {item.pattern}); artefact ({item.source}) "
                        f"gives {', '.join(item.expected)}")
            )
            continue
        if len(matches) < item.sites:
            findings.append(
                Finding("prose", item.label,
                        f"found this quantity at {len(matches)} site(s) but it is known to be "
                        f"stated at {item.sites}; a site has been reworded and is no longer "
                        f"being checked")
            )
        for found in matches:
            got = found if isinstance(found, tuple) else (found,)
            if got != item.expected:
                findings.append(
                    Finding("prose", item.label,
                            f"manuscript states {', '.join(got)} but artefact ({item.source}) "
                            f"gives {', '.join(item.expected)}")
                )
    return findings


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manuscript", required=True, type=Path)
    parser.add_argument("--runs-dir", required=True, type=Path,
                        help="acceptance_loop/runs directory holding the source runs")
    parser.add_argument("--hosted-run", required=True)
    parser.add_argument("--semantic-run", required=True)
    parser.add_argument("--e2e-run", required=True)
    parser.add_argument("--alternative-run", required=True)
    parser.add_argument("--clustered-run", required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="paper root holding src/ (default: parent of this script's directory)")
    parser.add_argument("--skip-exhibits", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    manuscript = args.manuscript.resolve()
    figures_dir = manuscript.parent / "Figures" / "acceptance_loop"
    runs = {
        "semantic": args.runs_dir / args.semantic_run,
        "hosted": args.runs_dir / args.hosted_run,
        "e2e": args.runs_dir / args.e2e_run,
        "alternative": args.runs_dir / args.alternative_run,
        "clustered": args.runs_dir / args.clustered_run,
    }
    for name, path in runs.items():
        if not path.is_dir():
            raise SystemExit(f"{name} run directory not found: {path}")

    hosted = runs["hosted"]
    assertions = build_assertions(
        _load_json(hosted / "metrics.json"), _load_json(hosted / "paired_displacement.json")
    )

    findings = check_prose(manuscript, assertions)
    if not args.skip_exhibits:
        generator = args.project / "src" / GENERATOR
        if not generator.exists():
            raise SystemExit(f"generator not found: {generator}")
        findings += check_exhibits(args.project, generator, runs, figures_dir)

    logger.info("manuscript : %s", manuscript)
    logger.info("hosted run : %s", hosted.name)
    logger.info("prose quantities checked: %d", len(assertions))
    if not findings:
        logger.info("RESULT: IN SYNC - every derivable quantity appears in the manuscript "
                    "and the shipped exhibits match a fresh regeneration")
        return 0
    logger.info("RESULT: DRIFT - %d finding(s)", len(findings))
    for f in findings:
        logger.info("  [%s] %s: %s", f.check, f.label, f.detail)
    return 1


if __name__ == "__main__":
    sys.exit(main())

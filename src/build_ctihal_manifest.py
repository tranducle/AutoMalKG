"""Build a pinned, observation-level inventory of the local CTI-HAL checkout.

This script audits the dataset; it does not reinterpret CTI-HAL's ATT&CK labels
as gold labels for the broader MALOnt schema.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiment_support import (
    DEFAULT_RUNS_ROOT,
    PAPER_ROOT,
    build_manifest,
    git_commit,
    resolve_java,
    utc_now,
    utc_run_stamp,
    write_json,
    write_jsonl,
)


PINNED_COMMIT = "fdf0678a153a17337266a2205293581ee333dfe9"


def _normalise_stem(stem: str, group: str) -> str:
    value = stem.lower()
    value = re.sub(rf"^{re.escape(group.lower())}[-_ ]*", "", value)
    return re.sub(r"[^a-z0-9]", "", value)


def _match_annotation_json(
    annotation_files: Iterable[Path], json_files: Iterable[Path], group: str
) -> dict[str, Any]:
    """Pair annotation and JSON names, tolerating documented abbreviations/typos."""

    annotations = {path: _normalise_stem(path.stem, group) for path in annotation_files}
    jsons = {path: _normalise_stem(path.stem, group) for path in json_files}
    matches: list[dict[str, Any]] = []
    unmatched_annotations = set(annotations)
    unmatched_json = set(jsons)

    for annotation in sorted(list(unmatched_annotations), key=str):
        exact = [path for path in unmatched_json if jsons[path] == annotations[annotation]]
        if exact:
            target = sorted(exact, key=str)[0]
            matches.append(
                {
                    "annotation": annotation.name,
                    "json": target.name,
                    "match": "normalized_exact",
                    "similarity": 1.0,
                }
            )
            unmatched_annotations.remove(annotation)
            unmatched_json.remove(target)

    candidates: list[tuple[float, str, str, Path, Path]] = []
    for annotation in unmatched_annotations:
        for target in unmatched_json:
            score = difflib.SequenceMatcher(
                None, annotations[annotation], jsons[target]
            ).ratio()
            candidates.append((score, str(annotation), str(target), annotation, target))
    for score, _, _, annotation, target in sorted(candidates, reverse=True):
        if annotation not in unmatched_annotations or target not in unmatched_json:
            continue
        # A conservative floor keeps unrelated names from being presented as matches.
        if score < 0.52:
            continue
        matches.append(
            {
                "annotation": annotation.name,
                "json": target.name,
                "match": "fuzzy_name",
                "similarity": round(score, 6),
            }
        )
        unmatched_annotations.remove(annotation)
        unmatched_json.remove(target)

    return {
        "matches": sorted(matches, key=lambda row: row["annotation"]),
        "unmatched_annotations": sorted(path.name for path in unmatched_annotations),
        "unmatched_json": sorted(path.name for path in unmatched_json),
    }


def _pdf_page_count(paths: Iterable[Path]) -> tuple[int | None, list[dict[str, str]]]:
    paths = list(paths)
    if not paths:
        return 0, []
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        return None, [{"stage": "pdf_pages", "error": "pdfinfo is not installed"}]
    total = 0
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            completed = subprocess.run(
                [pdfinfo, str(path)], check=False, capture_output=True, text=True, timeout=30
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or f"pdfinfo exit {completed.returncode}")
            page_line = next(
                line for line in completed.stdout.splitlines() if line.startswith("Pages:")
            )
            total += int(page_line.split(":", 1)[1].strip())
        except (OSError, RuntimeError, StopIteration, ValueError, subprocess.TimeoutExpired) as exc:
            failures.append({"stage": "pdf_pages", "path": str(path), "error": str(exc)})
    return total, failures


def inventory_ctihal(dataset_root: Path, expected_commit: str | None = PINNED_COMMIT) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return dataset inventory, per-file observations, and failures."""

    root = dataset_root.resolve()
    json_files = sorted((root / "data").rglob("*.json"))
    annotation_files = sorted((root / "annotations").rglob("*.md"))
    report_pdfs = sorted((root / "reports").rglob("*.pdf"))
    annotated_pdfs = sorted((root / "annotated reports").rglob("*.pdf"))
    failures: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    row_fingerprints: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    context_labels: dict[str, set[str]] = defaultdict(set)
    techniques: set[str] = set()
    tactics: set[str] = set()
    rows_total = 0
    labeled_rows = 0
    group_counts: Counter[str] = Counter()

    commit = git_commit(root)
    if expected_commit and commit != expected_commit:
        failures.append(
            {
                "stage": "commit_pin",
                "expected": expected_commit,
                "observed": commit,
                "error": "CTI-HAL checkout is not at the preregistered commit",
            }
        )

    for path in json_files:
        relative = path.relative_to(root)
        group = relative.parts[1] if len(relative.parts) > 2 else relative.parts[0]
        group_counts[group] += 1
        record: dict[str, Any] = {
            "case_id": f"ctihal-json-{len(observations):03d}",
            "path": relative.as_posix(),
            "group": group,
            "parse_status": "VALID",
            "row_count": 0,
            "labeled_row_count": 0,
        }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise TypeError(f"top-level value is {type(payload).__name__}, expected list")
            record["row_count"] = len(payload)
            rows_total += len(payload)
            for row_index, row in enumerate(payload):
                if not isinstance(row, dict):
                    failures.append(
                        {
                            "stage": "row_schema",
                            "path": relative.as_posix(),
                            "row_index": row_index,
                            "error": "row is not an object",
                        }
                    )
                    continue
                technique = row.get("technique")
                if technique is not None:
                    labeled_rows += 1
                    record["labeled_row_count"] += 1
                    techniques.add(str(technique))
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                for tactic in metadata.get("tactic", []) or []:
                    tactics.add(str(tactic))
                fingerprint = json.dumps(row, sort_keys=True, ensure_ascii=False)
                row_fingerprints[fingerprint] += 1
                context = str(row.get("context", "")).strip()
                if context:
                    context_counts[context] += 1
                    context_labels[context].add(str(technique))
        except Exception as exc:
            record["parse_status"] = "INVALID"
            record["error"] = str(exc)
            failures.append(
                {"stage": "json_parse", "path": relative.as_posix(), "error": str(exc)}
            )
        observations.append(record)

    report_pages, page_failures = _pdf_page_count(report_pdfs)
    failures.extend(page_failures)

    license_files = sorted(
        {
            path
            for pattern in ("LICENSE*", "COPYING*", "NOTICE*")
            for path in root.rglob(pattern)
            if path.is_file()
        },
        key=str,
    )

    conversion_audit: dict[str, Any] = {}
    groups = sorted(
        {path.relative_to(root / "annotations").parts[0] for path in annotation_files}
    )
    for group in groups:
        if group == "apt29":
            continue
        conversion_audit[group] = _match_annotation_json(
            (root / "annotations" / group).glob("*.md"),
            (root / "data" / group).glob("*.json"),
            group,
        )

    apt29_by_annotator: dict[str, set[str]] = {}
    apt29_root = root / "data" / "apt29"
    for annotator_dir in sorted(path for path in apt29_root.iterdir() if path.is_dir()):
        apt29_by_annotator[annotator_dir.name] = {
            path.stem.upper() for path in annotator_dir.glob("*.json")
        }
    paired_stems = (
        set.intersection(*apt29_by_annotator.values()) if apt29_by_annotator else set()
    )

    inventory = {
        "dataset": "CTI-HAL",
        "dataset_root": str(root),
        "git_commit": commit,
        "expected_git_commit": expected_commit,
        "commit_matches_pin": commit == expected_commit if expected_commit else None,
        "counts": {
            "canonical_report_pdfs": len(report_pdfs),
            "canonical_report_pages": report_pages,
            "annotated_report_pdfs": len(annotated_pdfs),
            "annotation_markdown_files": len(annotation_files),
            "json_files": len(json_files),
            "json_rows": rows_total,
            "labeled_technique_rows": labeled_rows,
            "unlabeled_technique_rows": rows_total - labeled_rows,
            "unique_techniques": len(techniques),
            "unique_raw_tactic_values": len(tactics),
        },
        "json_files_by_group": dict(sorted(group_counts.items())),
        "duplicate_audit": {
            "exact_duplicate_row_excess": sum(count - 1 for count in row_fingerprints.values() if count > 1),
            "exact_duplicate_row_groups": sum(1 for count in row_fingerprints.values() if count > 1),
            "repeated_contexts": sum(1 for count in context_counts.values() if count > 1),
            "contexts_with_multiple_technique_values": sum(1 for labels in context_labels.values() if len(labels) > 1),
            "note": "Duplicate rows/contexts may include legitimate parallel annotation and are not silently removed.",
        },
        "annotator_audit": {
            "apt29_annotator_file_counts": {
                name: len(stems) for name, stems in sorted(apt29_by_annotator.items())
            },
            "apt29_independently_double_annotated_pairs": len(paired_stems),
            "apt29_paired_report_stems": sorted(paired_stems),
        },
        "annotation_to_json_conversion_audit": conversion_audit,
        "license_audit": {
            "detected_files": [path.relative_to(root).as_posix() for path in license_files],
            "repository_license_detected": bool(license_files),
            "caveat": (
                "No LICENSE/COPYING/NOTICE file was detected. Do not redistribute source files "
                "until terms are confirmed; publish only derived inventories/hashes."
                if not license_files
                else "Detected files must still be reviewed for dataset redistribution terms."
            ),
        },
        "task_fit_caveat": (
            "CTI-HAL provides ATT&CK technique/context annotations. It is not gold ground truth "
            "for AutoMalKG's full ThreatActor/Malware/Vulnerability/Campaign/Indicator schema, "
            "so this inventory cannot license full-schema entity/relation accuracy claims."
        ),
        "raw_unique_tactic_values": sorted(tactics),
    }
    return inventory, observations, failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PAPER_ROOT / "data" / "raw" / "CTI-HAL",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-commit", default=PINNED_COMMIT)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started_at = utc_now()
    run_id = f"ctihal_inventory_{utc_run_stamp()}"
    output_dir = (args.output_dir or DEFAULT_RUNS_ROOT / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory, observations, failures = inventory_ctihal(
        args.dataset_root, expected_commit=args.expected_commit
    )

    inventory_path = output_dir / "ctihal_inventory.json"
    cases_path = output_dir / "cases.jsonl"
    failures_path = output_dir / "failures.jsonl"
    readme_path = output_dir / "README.md"
    write_json(inventory_path, inventory)
    write_jsonl(cases_path, observations)
    write_jsonl(failures_path, failures)
    replay_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--output-dir",
        str(output_dir),
        "--expected-commit",
        args.expected_commit,
        "--seed",
        str(args.seed),
    ]
    readme_path.write_text(
        "# CTI-HAL inventory run\n\n"
        "This run inventories the pinned local checkout. It does not evaluate full MALOnt extraction accuracy.\n\n"
        f"Replay: `{' '.join(replay_command)}`\n\n"
        "License caveat: no repository LICENSE/COPYING/NOTICE file was detected at the pinned revision; "
        "do not redistribute CTI-HAL source files without confirming terms.\n",
        encoding="utf-8",
    )
    finished_at = utc_now()
    _, java_audit = resolve_java()
    input_files = sorted(args.dataset_root.resolve().rglob("*"))
    output_files = [inventory_path, cases_path, failures_path, readme_path]
    manifest = build_manifest(
        run_id=run_id,
        command=[sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        seed=args.seed,
        started_at=started_at,
        finished_at=finished_at,
        inputs=input_files,
        outputs=output_files,
        base=PAPER_ROOT,
        java=java_audit,
        dependencies=(),
        extra={
            "experiment_id": "E4-data-provenance",
            "dataset_git_commit": inventory["git_commit"],
            "expected_dataset_git_commit": args.expected_commit,
            "gate_interpretation": "PASS_WITH_WARNINGS" if not failures else "FAIL_REPAIR",
        },
    )
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "counts": inventory["counts"], "failures": len(failures)}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

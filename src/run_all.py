"""Export legacy aggregate summaries as explicitly non-evidentiary archives.

This command does *not* run experiments or recompute metrics.  It wraps the
existing aggregate JSON files with provenance and a hard warning that the
payload cannot support empirical claims without item-level source artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "data" / "archive_exports"
ARCHIVE_FILES = {
    "exp1": "exp1_results.json",
    "exp2": "exp2_results.json",
    "exp3": "exp3_results.json",
    "exp4": "exp4_results.json",
    "exp5": "exp5_results.json",
}
NON_EVIDENCE_STATUS = "non_evidence_unverifiable_aggregate"
NON_EVIDENCE_WARNING = (
    "Archive export only: item-level labels, predictions, run logs, and/or "
    "source records are unavailable. The archived payload is not a rerun and "
    "must not be used as evidence for quantitative or comparative claims."
)


class ArchiveExportError(RuntimeError):
    """Raised when a legacy archive cannot be exported safely."""


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def export_archive(
    experiment_id: str,
    *,
    source_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Wrap one stored aggregate in a deterministic non-evidence envelope."""

    if experiment_id not in ARCHIVE_FILES:
        raise ArchiveExportError(f"unknown legacy experiment ID: {experiment_id}")
    source = Path(source_path or PROCESSED_DIR / ARCHIVE_FILES[experiment_id])
    output = Path(
        output_path
        or DEFAULT_EXPORT_DIR / f"{experiment_id}_archived_summary.non_evidence.json"
    )
    if not source.is_file():
        raise ArchiveExportError(
            f"archived source is missing; no experiment was run: {_display_path(source)}"
        )
    if source.resolve() == output.resolve():
        raise ArchiveExportError("archive export must not overwrite its source")

    raw = source.read_bytes()
    try:
        archived_payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ArchiveExportError(f"archived source is not valid JSON: {exc}") from exc
    if not isinstance(archived_payload, dict):
        raise ArchiveExportError("archived source root must be a JSON object")

    envelope = {
        "artifact_type": "legacy_archive_export",
        "evidence_status": NON_EVIDENCE_STATUS,
        "experiment_id": experiment_id,
        "warning": NON_EVIDENCE_WARNING,
        "source": {
            "path": _display_path(source),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "archived_payload": archived_payload,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def export_all(
    experiment_ids: Iterable[str] = ARCHIVE_FILES,
    *,
    output_dir: Path = DEFAULT_EXPORT_DIR,
) -> list[Path]:
    outputs = []
    for experiment_id in experiment_ids:
        outputs.append(
            export_archive(
                experiment_id,
                output_path=Path(output_dir)
                / f"{experiment_id}_archived_summary.non_evidence.json",
            )
        )
    return outputs


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export stored legacy aggregates with a non-evidence warning."
    )
    parser.add_argument(
        "--experiment",
        action="append",
        choices=sorted(ARCHIVE_FILES),
        help="Legacy archive to export; repeat for multiple. Defaults to all.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    args = parser.parse_args(argv)

    print("=== AutoMalKG legacy archive export (NOT experiment execution) ===")
    print(NON_EVIDENCE_WARNING)
    for path in export_all(args.experiment or ARCHIVE_FILES, output_dir=args.output_dir):
        print(f"exported: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

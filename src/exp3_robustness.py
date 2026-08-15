"""Export the legacy robustness aggregate; this does not run Experiment 3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

try:
    from .run_all import export_archive
except ImportError:  # pragma: no cover - script entry point.
    from run_all import export_archive


def run_exp3(source: Optional[Path] = None, output: Optional[Path] = None) -> Path:
    """Compatibility wrapper that produces an explicit non-evidence archive."""

    return export_archive("exp3", source_path=source, output_path=output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print("NOT AN EXPERIMENT RUN: exporting an unverifiable aggregate archive.")
    print(run_exp3(args.source, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

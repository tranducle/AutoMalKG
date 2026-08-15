#!/usr/bin/env python3
"""Verify an extracted AutoMalKG supplement against MANIFEST.sha256."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path, PurePosixPath

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = root / "MANIFEST.sha256"
    failures = []
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        member = PurePosixPath(relative)
        if member.is_absolute() or ".." in member.parts:
            failures.append(f"line {number}: unsafe path")
            continue
        path = root.joinpath(*member.parts)
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif digest(path) != expected:
            failures.append(f"hash mismatch: {relative}")
    if failures:
        print("FAIL")
        print("\n".join(failures))
        return 1
    print("PASS: internal manifest verified")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

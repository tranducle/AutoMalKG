"""Shared reproducibility utilities for AutoMalKG experiment harnesses.

The helpers in this module deliberately resolve paths from this file instead of
the caller's working directory.  A run manifest hashes every material input and
output artifact; ``manifest.json`` itself is excluded because a file cannot
contain its own stable cryptographic digest.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PAPER_ROOT / "reproduced_runs"


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def utc_run_stamp() -> str:
    """Return a filesystem-safe UTC run stamp."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_files(paths: Iterable[Path], *, base: Path | None = None) -> list[dict[str, Any]]:
    """Return deterministic metadata and hashes for existing regular files."""

    records: list[dict[str, Any]] = []
    for raw_path in sorted({Path(path).resolve() for path in paths}, key=str):
        if not raw_path.is_file():
            continue
        try:
            display = str(raw_path.relative_to(base.resolve())) if base else str(raw_path)
        except ValueError:
            display = str(raw_path)
        records.append(
            {
                "path": display,
                "bytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
            }
        )
    return records


def write_json(path: Path, value: Any) -> None:
    """Write stable, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write one canonical JSON object per line."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False))
            handle.write("\n")


def command_output(command: Sequence[str]) -> dict[str, Any]:
    """Run a version/provenance command and retain its truthful outcome."""

    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": shlex.join(command), "status": "unavailable", "error": str(exc)}
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    return {
        "command": shlex.join(command),
        "status": "ok" if completed.returncode == 0 else "error",
        "returncode": completed.returncode,
        "output": output,
    }


def resolve_java(explicit: str | Path | None = None) -> tuple[Path | None, dict[str, Any]]:
    """Find a functioning Java executable, rejecting the macOS GUI stub."""

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidates.append(Path(java_home) / "bin" / "java")
    candidates.extend(
        [
            Path("/opt/homebrew/opt/openjdk/bin/java"),
            Path("/usr/local/opt/openjdk/bin/java"),
        ]
    )
    on_path = shutil.which("java")
    if on_path:
        candidates.append(Path(on_path))

    seen: set[Path] = set()
    attempts: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        result = command_output([str(candidate), "-version"])
        attempts.append({"path": str(candidate), **result})
        if result.get("status") == "ok":
            return candidate, {"selected": str(candidate), "attempts": attempts}
    return None, {"selected": None, "attempts": attempts}


def dependency_versions(names: Sequence[str]) -> dict[str, str | None]:
    """Resolve installed distribution versions without importing packages."""

    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit(path: Path) -> str | None:
    """Return the commit containing *path*, or ``None`` outside a Git checkout."""

    result = command_output(["git", "-C", str(path), "rev-parse", "HEAD"])
    if result.get("status") != "ok":
        return None
    return str(result.get("output", "")).splitlines()[0].strip() or None


def hardware_summary() -> dict[str, Any]:
    """Collect a dependency-free host summary suitable for a run manifest."""

    memory_bytes: int | None = None
    try:
        memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        pass
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_bytes": memory_bytes,
    }


def build_manifest(
    *,
    run_id: str,
    command: Sequence[str],
    seed: int,
    started_at: str,
    finished_at: str,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    base: Path,
    java: Mapping[str, Any] | None = None,
    dependencies: Sequence[str] = ("owlready2",),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common observation-level run manifest."""

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "command": shlex.join(command),
        "seed": seed,
        "timestamps": {"started_at": started_at, "finished_at": finished_at},
        "project_git_commit": git_commit(PAPER_ROOT),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "dependencies": dependency_versions(dependencies),
        "java": dict(java or {}),
        "os_hardware": hardware_summary(),
        "provider": None,
        "resolved_model": None,
        "inputs": hash_files(inputs, base=base),
        "outputs": hash_files(outputs, base=base),
        "hash_policy": {
            "algorithm": "sha256",
            "manifest_self_hash": "excluded_by_definition_to_avoid_self-reference",
        },
    }
    if extra:
        manifest.update(dict(extra))
    return manifest


def quantile(values: Sequence[float], probability: float) -> float | None:
    """Compute a deterministic linearly interpolated sample quantile."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

#!/usr/bin/env python3
"""Build deterministic, anonymous AutoMalKG supplementary archives.

The builder uses explicit allowlists.  It never packages CTI-HAL source text,
hosted raw responses, hosted predictions, populated CTI ontologies, or replay
diagnostics.  Text artifacts are sanitized before packaging, every member is
hashed in an internal manifest, and tar/gzip metadata are fixed so identical
inputs produce byte-identical archives.

Usage:
    python3 src/build_anonymous_supplements.py
    python3 src/build_anonymous_supplements.py --check-reproducible
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


ARCHIVE_MTIME = 0
ARCHIVE_MODE = 0o644
EXECUTABLE_MODE = 0o755
REPRODUCTION_REQUIREMENTS = b"pytest==9.0.2\nscipy==1.17.1\n"
S1_FILENAME = "AutoMalKG_S1_reproducible_core_v2.tar.gz"
S2_FILENAME = "AutoMalKG_S2_hosted_audit_public_v2.tar.gz"

RUNS_REL = Path("reproduced_runs")
SEMANTIC_RUN = "semantic_benchmark_seed20260809_comprehensive"
REACHABILITY_RUN = "offline_stage_reachability_fixture_only_20260809_v6"
HOSTED_RUN = "hosted_round3_cc0_20260810T004042Z"
REPLAY_RUN = "hosted_end_to_end_replay_20260809_round3_v2"
CLUSTER_RUN = "hosted_clustered_reanalysis_20260809"
ROUND3_BENCHMARK_REL = Path("benchmark/round3_cc0")

TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".owl",
    ".py",
    ".txt",
}

FORBIDDEN_MEMBER_PARTS = {
    "hosted_run/cases.jsonl",
    "hosted_run/predictions.jsonl",
    "hosted_run/raw_responses.jsonl",
    "hosted_run/failures.jsonl",
    "cluster_analysis/injection_pairs.jsonl",
    "case_artifacts",
    "data/raw",
    ".env",
}

SECRET_PATTERNS = {
    "openai_style_key": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(rb"gh[oprsu]_[A-Za-z0-9]{20,}"),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "assigned_api_key": re.compile(
        rb"(?i)(?:api[_-]?key|token|secret)\s*[=:]\s*['\"]?[A-Za-z0-9_./+-]{24,}"
    ),
}


@dataclass(frozen=True)
class PayloadFile:
    """One archive member and its provenance."""

    archive_path: str
    data: bytes
    source: str
    source_sha256: str | None
    transformed: bool
    mode: int = ARCHIVE_MODE


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def stable_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(dict(row), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def validate_archive_path(path: str) -> str:
    posix = PurePosixPath(path)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise ValueError(f"unsafe archive member path: {path!r}")
    normalized = posix.as_posix()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def sanitize_text(text: str, *, paper_root: Path) -> tuple[str, bool]:
    """Remove machine-specific absolute paths without altering scientific values."""

    original = text
    paper_absolute = paper_root.resolve().as_posix()
    repository_absolute = paper_root.resolve().parents[1].as_posix()
    text = text.replace(paper_absolute, ".")
    text = text.replace(repository_absolute, "<REPOSITORY_ROOT>")

    # These patterns intentionally target machine paths, not URLs or ontology IRIs.
    users_prefix = "/" + "Users/"
    home_prefix = "/" + "home/"
    var_folders_prefix = "/" + "var/folders/"
    private_var_prefix = "/" + "private/var/"
    replacements = (
        (re.compile(re.escape(users_prefix) + r"[^\s`\"'<>]+"), "<LOCAL_PATH>"),
        (re.compile(re.escape(home_prefix) + r"[^\s`\"'<>]+"), "<LOCAL_PATH>"),
        (re.compile(re.escape(var_folders_prefix) + r"[^\s`\"'<>]+"), "<TEMP_PATH>"),
        (re.compile(re.escape(private_var_prefix) + r"[^\s`\"'<>]+"), "<TEMP_PATH>"),
    )
    for pattern, replacement in replacements:
        text = pattern.sub(replacement, text)
    # Test fixtures sometimes use credential-shaped sentinels.  They prove
    # redaction behavior but must not be shipped in credential shape.
    text = re.sub(r"sk-[A-Za-z0-9_-]{20,}", "<REDACTED_TOKEN>", text)
    text = re.sub(
        r"(?i)\b(api[_-]?key|token|secret)\s*=\s*(['\"])[^'\"]{20,}\2",
        lambda match: f'{match.group(1)} = "<REDACTED_TOKEN>"',
        text,
    )
    return text, text != original


def payload_from_file(
    paper_root: Path,
    source_relative: Path,
    archive_path: str,
    *,
    sanitize: bool = True,
) -> PayloadFile:
    source = paper_root / source_relative
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = source.read_bytes()
    transformed = False
    data = raw
    if sanitize and source.suffix.lower() in TEXT_SUFFIXES:
        decoded = raw.decode("utf-8")
        cleaned, transformed = sanitize_text(decoded, paper_root=paper_root)
        data = cleaned.encode("utf-8")
    return PayloadFile(
        archive_path=validate_archive_path(archive_path),
        data=data,
        source=source_relative.as_posix(),
        source_sha256=sha256_bytes(raw),
        transformed=transformed,
    )


def generated_payload(archive_path: str, data: bytes, *, mode: int = ARCHIVE_MODE) -> PayloadFile:
    return PayloadFile(
        archive_path=validate_archive_path(archive_path),
        data=data,
        source="generated",
        source_sha256=None,
        transformed=False,
        mode=mode,
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _export_schema_ontology(paper_root: Path) -> bytes:
    """Export the code-defined schema without loading any empirical instances."""

    src_dir = paper_root / "src"
    sys.path.insert(0, str(src_dir))
    try:
        from reasoning import MalOntReasoner  # type: ignore

        with tempfile.TemporaryDirectory(prefix="automalkg-schema-") as temporary:
            target = Path(temporary) / "automalkg_schema.owl"
            reasoner = MalOntReasoner("https://example.org/automalkg/schema#")
            reasoner.onto.save(file=str(target), format="rdfxml")
            return target.read_bytes()
    finally:
        try:
            sys.path.remove(str(src_dir))
        except ValueError:
            pass


INTERNAL_VERIFIER = r'''#!/usr/bin/env python3
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
'''.encode("utf-8")


OUTER_VERIFIER = r'''#!/usr/bin/env python3
"""Verify AutoMalKG supplement archives and their internal manifests."""
from __future__ import annotations
import argparse
import hashlib
import io
import tarfile
from pathlib import Path, PurePosixPath

def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def verify_archive(path: Path) -> list[str]:
    failures = []
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if names != sorted(names):
            failures.append(f"{path.name}: members are not sorted")
        if len(names) != len(set(names)):
            failures.append(f"{path.name}: duplicate members")
        for member in members:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or not member.isfile():
                failures.append(f"{path.name}: unsafe member {member.name}")
            if member.mtime != 0 or member.uid != 0 or member.gid != 0:
                failures.append(f"{path.name}: non-canonical metadata {member.name}")
        manifest_member = archive.getmember("MANIFEST.sha256")
        manifest = archive.extractfile(manifest_member)
        assert manifest is not None
        for line in io.TextIOWrapper(manifest, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                continue
            expected, name = line.split("  ", 1)
            try:
                member = archive.getmember(name)
            except KeyError:
                failures.append(f"{path.name}: missing {name}")
                continue
            stream = archive.extractfile(member)
            assert stream is not None
            if digest_bytes(stream.read()) != expected:
                failures.append(f"{path.name}: internal hash mismatch {name}")
    return failures

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default=".")
    args = parser.parse_args()
    root = Path(args.directory)
    failures = []
    checksum_path = root / "SHA256SUMS"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split("  ", 1)
        archive_path = root / name
        if not archive_path.is_file():
            failures.append(f"missing archive: {name}")
            continue
        if digest_file(archive_path) != expected:
            failures.append(f"archive hash mismatch: {name}")
            continue
        failures.extend(verify_archive(archive_path))
    if failures:
        print("FAIL")
        print("\n".join(failures))
        return 1
    print("PASS: archive and internal manifests verified")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''.encode("utf-8")


def _core_readme() -> bytes:
    return b"""# AutoMalKG S1 reproducible core

This anonymous package contains the executable Python source, tests, pinned
direct dependency constraints, the code-exported OWL/SWRL schema, and the
synthetic semantic/reachability evidence used by the manuscript.  It also
contains the author-created CC0 Round-3 benchmark required by its offline tests;
the benchmark provenance statement and CC0 dedication are included in place.

It contains no CTI-HAL source text, hosted responses, hosted predictions,
credentials, populated CTI ontology, or personal absolute paths.  The
reachability package contains all 26 synthetic fixtures and their observations;
frozen hosted-response observations are intentionally excluded and summarized
only in S2.

Requirements: Python 3.12+ and a functioning Java runtime on PATH for Pellet.

Quick checks from the extracted package root:

```bash
python3 verify_archive.py .
python3 -m pip install -r requirements.txt -r requirements-reproduction.txt
python3 -m pytest -q
python3 src/run_semantic_benchmark.py --output-dir reproduced_runs/semantic-minimal --seed 20260809 --consistency-cases-per-class 1 --swrl-controls-per-class 1 --scale-sizes 10 --scale-repeats 1
python3 src/run_stage_reachability_ablation.py --fixtures-only --output-dir reproduced_runs/reachability
```

The runners refuse to overwrite an existing output directory.  Runtime values
are environment-sensitive; contract/status outcomes are the reproducibility
target.  The fixture-only replay reads no hosted cases or raw responses and
emits 29 synthetic contract observations plus the OWL/SWRL/fault-injection
reachability ledger.  `PACKAGE_METADATA.json` records the source hash and
packaged hash for every member.  Executable builder and replay-runner source is
packaged byte-for-byte; machine-path sanitization is limited to non-executable
evidence where needed.
"""


def _hosted_readme() -> bytes:
    return b"""# AutoMalKG S2 public hosted audit

This anonymous package is a public-safe audit projection of the frozen hosted
run.  It includes the exact system prompts and user-prompt construction
template, executed runner snapshot, sanitized run configuration and manifests,
aggregate metrics, scientific gates, source/output hashes, and a pseudonymized
case-level stage ledger.

## Contents

| Path | Contents |
| --- | --- |
| `prompts.json` | System prompt and user-prompt construction template. |
| `hosted_run/` | The reported run `hosted_round3_cc0_20260810T004042Z`: executed runner snapshot, sanitized run configuration, pre-call manifest, aggregate metrics, paired-displacement analysis, pricing snapshot, scientific gates, and pre/post credential scans. |
| `offline_replay/` | Frozen-response end-to-end replay: metrics, integrity gates, manifest, and the projected per-call stage ledger. |
| `cluster_analysis/` | **Superseded, retained as a record.** Offline clustered reanalysis of the earlier 126-call hosted stress run. No result reported in the manuscript comes from it; see `cluster_analysis/README.md`. |
| `MANIFEST.sha256`, `verify_archive.py` | SHA-256 manifest of this package and its verifier. |
| `PACKAGE_METADATA.json` | Per-member packaged and source digests. |

## What is withheld, and what is not

Withheld from this package: raw model responses, free-text predictions,
`cases.jsonl`, validation diagnostics that reproduce model-generated entity
text, injection-pair entity sets, request identifiers, populated ontologies,
and case-level logs and artifacts.  The stage ledger's SHA-256 digests
establish response identity without republishing the text.

**The evaluation corpus is not withheld.**  The reported hosted run uses 16
author-authored contexts dedicated under CC0-1.0, and they are published in
full -- with gold annotations, manifest, design ledger and dedication -- at
`benchmark/round3_cc0/` in the artifact repository and at
`5_Data_and_Experiments/acceptance_loop/round3_cc0_benchmark_v2/` in supplement
S1.  Nothing here is held back for want of a licence to the evaluation
contexts.

The one third-party corpus this package does not carry is CTI-HAL, which the
manuscript uses for related work and which supplied the contexts of the
superseded stress run projected in `cluster_analysis/`.  Third-party licensing
for hosted processing of CTI-HAL was not established, so CTI-HAL text is not
redistributed here; hashes and source coordinates remain the audit bridge to an
independently obtained pinned checkout.  This bears on the superseded
reanalysis only, not on the reported round-3 run.

The projected ledger preserves condition, repeat, source/response hashes,
pipeline stage statuses, scalar entity/relation counts, runtime, and no-hosted-
call evidence.  `prompts.json` uses `{{CTI_CONTEXT}}` as the context redaction
and `{{INJECTION_NOTE}}` as the injection-note redaction;
`executed_run_hosted_round3_cc0.py` is the exact historical prompt constructor.
`paired_displacement.json` reports the paired clean-vs-injected displacement
analysis (status, canonical graph, gold-score) for the 48 matched pairs.

Verify an extracted package with:

```bash
python3 verify_archive.py .
```

These data measure response-contract behavior and offline pipeline attrition,
not factual extraction accuracy.
"""


CLUSTER_SUPERSESSION_NOTE = b"""> **Superseded -- retained as a record.**
>
> This reanalysis belongs to the earlier hosted stress run
> `hosted_stress_20260809T211342Z`: 126 calls over 14 sampled CTI-HAL contexts
> in 7 groups.  It is superseded by `hosted_round3_cc0_20260810T004042Z`, the
> run projected in `../hosted_run/`: 144 calls over 16 author-authored CC0-1.0
> contexts.  Round 3 replaced the CTI-HAL sample with a corpus whose licensing
> permits publication, and it is the run every hosted result in the manuscript
> reports.  No number below is reported in the manuscript.  The reanalysis is
> retained so that the earlier analysis, and the change of corpus, stay
> auditable.

"""


def _projected_cluster_readme(paper_root: Path) -> PayloadFile:
    """Label the superseded reanalysis in the projection, not at its source.

    The run directory's own README is covered by that run's `manifest.json`, so
    editing it there would break the frozen run's integrity record.  The
    supersession note is prepended to the public copy instead; the source digest
    in `PACKAGE_METADATA.json` still identifies the unedited original.
    """

    source_relative = RUNS_REL / CLUSTER_RUN / "README.md"
    projected = payload_from_file(
        paper_root, source_relative, "cluster_analysis/README.md"
    )
    return replace(projected, data=CLUSTER_SUPERSESSION_NOTE + projected.data)


def _package_metadata(package_id: str, files: Iterable[PayloadFile]) -> bytes:
    entries = []
    for item in sorted(files, key=lambda value: value.archive_path):
        entries.append(
            {
                "archive_path": item.archive_path,
                "bytes": len(item.data),
                "packaged_sha256": sha256_bytes(item.data),
                "source": item.source,
                "source_sha256": item.source_sha256,
                "transformed_for_anonymity": item.transformed,
            }
        )
    return stable_json(
        {
            "archive_schema": "automalkg-anonymous-supplement-v2",
            "package_id": package_id,
            "archive_metadata": {
                "mtime": ARCHIVE_MTIME,
                "uid": 0,
                "gid": 0,
                "uname": "",
                "gname": "",
                "file_mode": oct(ARCHIVE_MODE),
            },
            "hash_algorithm": "sha256",
            "members": entries,
            "scope_note": (
                "source_sha256 identifies the local evidence file before path "
                "sanitization; packaged_sha256 verifies the public-safe copy. "
                "cluster_analysis/README.md is its source README with the "
                "supersession note prepended, so its two digests differ by that "
                "note alone."
            ),
        }
    )


def _finalize_payload(package_id: str, files: list[PayloadFile], readme: bytes) -> list[PayloadFile]:
    paths = [item.archive_path for item in files]
    if len(paths) != len(set(paths)):
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        raise ValueError(f"duplicate archive paths: {duplicates}")

    files = list(files)
    files.append(generated_payload("README.md", readme))
    files.append(generated_payload("verify_archive.py", INTERNAL_VERIFIER, mode=EXECUTABLE_MODE))
    files.append(generated_payload("PACKAGE_METADATA.json", _package_metadata(package_id, files)))
    manifest = "".join(
        f"{sha256_bytes(item.data)}  {item.archive_path}\n"
        for item in sorted(files, key=lambda value: value.archive_path)
    ).encode("utf-8")
    files.append(generated_payload("MANIFEST.sha256", manifest))
    return sorted(files, key=lambda value: value.archive_path)


def _public_round3_benchmark_payload(paper_root: Path) -> list[PayloadFile]:
    """Sanitize the CC0 bundle and rebuild its nested content manifest."""

    member_names = (
        "contexts.jsonl",
        "gate_G0_data_provenance.json",
        "author_source.json",
        "CC0-1.0-DEDICATION.md",
        "DESIGN_LEDGER.md",
    )
    members: dict[str, PayloadFile] = {}
    for name in member_names:
        source_relative = ROUND3_BENCHMARK_REL / name
        members[name] = payload_from_file(
            paper_root,
            source_relative,
            source_relative.as_posix(),
        )

    manifest_relative = ROUND3_BENCHMARK_REL / "benchmark_manifest.json"
    manifest_source = paper_root / manifest_relative
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    expected_names = set(manifest.get("files", {}))
    if expected_names != set(member_names):
        raise ValueError(
            "Round-3 benchmark manifest member drift: "
            f"expected {sorted(member_names)}, found {sorted(expected_names)}"
        )
    for name, payload in members.items():
        manifest["files"][name] = {
            "bytes": len(payload.data),
            "sha256": sha256_bytes(payload.data),
        }
    manifest_data = stable_json(manifest)
    members["benchmark_manifest.json"] = PayloadFile(
        archive_path=manifest_relative.as_posix(),
        data=manifest_data,
        source=manifest_relative.as_posix(),
        source_sha256=sha256_file(manifest_source),
        transformed=manifest_data != manifest_source.read_bytes(),
    )
    return [members[name] for name in sorted(members)]


def build_s1_payload(paper_root: Path) -> list[PayloadFile]:
    files: list[PayloadFile] = []
    for directory in (Path("src"), Path("tests")):
        for source in sorted((paper_root / directory).glob("*.py")):
            relative = source.relative_to(paper_root)
            files.append(payload_from_file(paper_root, relative, relative.as_posix()))
    files.append(payload_from_file(paper_root, Path("requirements.txt"), "requirements.txt"))
    files.append(
        generated_payload(
            "requirements-reproduction.txt", REPRODUCTION_REQUIREMENTS
        )
    )
    files.append(
        generated_payload("ontology/automalkg_schema.owl", _export_schema_ontology(paper_root))
    )
    files.append(
        generated_payload(
            "ontology/README.md",
            b"The schema is exported from src/reasoning.py without empirical individuals.\n",
        )
    )

    files.extend(_public_round3_benchmark_payload(paper_root))

    semantic_rel = RUNS_REL / SEMANTIC_RUN
    semantic_names = [
        "README.md",
        "cases.jsonl",
        "predictions.jsonl",
        "metrics.json",
        "failures.jsonl",
        "G3_method_mechanism.json",
        "manifest.json",
    ]
    semantic_root = paper_root / semantic_rel
    semantic_names.extend(
        path.relative_to(semantic_root).as_posix()
        for path in sorted((semantic_root / "pellet_logs").glob("*.log"))
    )
    for name in semantic_names:
        files.append(
            payload_from_file(
                paper_root,
                semantic_rel / name,
                f"evidence/semantic/{name}",
            )
        )

    reach_rel = RUNS_REL / REACHABILITY_RUN
    for name in (
        "README.md",
        "fixtures.jsonl",
        "observations.jsonl",
        "stage_reachability_matrix.jsonl",
        "metrics.json",
        "G1_input_parity.json",
        "G2_baseline_sanity.json",
        "G4_ablation_validity.json",
        "G6_result_integrity.json",
        "baseline_fairness_audit.json",
        "manifest.json",
        "pip_freeze.txt",
        "executed_run_stage_reachability_ablation.py",
    ):
        files.append(
            payload_from_file(
                paper_root,
                reach_rel / name,
                f"evidence/reachability/{name}",
            )
        )

    observations = _load_jsonl(paper_root / reach_rel / "observations.jsonl")
    synthetic = [row for row in observations if row.get("source_kind") == "synthetic_fixture"]
    if len(synthetic) != 29 or len(synthetic) != len(observations):
        raise ValueError(
            "fixture-only reachability evidence must contain exactly 29 synthetic "
            f"observations and no hosted rows; found {len(synthetic)}/{len(observations)}"
        )
    files.append(
        generated_payload(
            "evidence/reachability/synthetic_observations.jsonl", stable_jsonl(synthetic)
        )
    )

    reach_logs = paper_root / reach_rel / "pellet_logs"
    for pattern in ("fixture-*.log", "owl-*.log", "swrl-*.log"):
        for source in sorted(reach_logs.glob(pattern)):
            relative = source.relative_to(paper_root)
            files.append(
                payload_from_file(
                    paper_root,
                    relative,
                    f"evidence/reachability/pellet_logs/{source.name}",
                )
            )
    return _finalize_payload("S1-reproducible-core-v2", files, _core_readme())


def _prompt_payload(paper_root: Path) -> bytes:
    src_dir = paper_root / "src"
    sys.path.insert(0, str(src_dir))
    try:
        import run_hosted_round3_cc0 as hosted  # type: ignore

        templates = {}
        placeholder = "{{CTI_CONTEXT}}"
        injection_placeholder = "{{INJECTION_NOTE}}"
        for condition in hosted.CONDITIONS:
            system, user = hosted._prompt_for(condition, placeholder, injection_placeholder)
            templates[condition] = {
                "system_prompt": system,
                "system_prompt_sha256": sha256_bytes(system.encode("utf-8")),
                "user_prompt_template": user,
                "user_prompt_template_sha256": sha256_bytes(user.encode("utf-8")),
            }
        return stable_json(
            {
                "schema_version": "1.0",
                "context_redaction_token": placeholder,
                "injection_note_redaction_token": injection_placeholder,
                "construction": "Replace {{CTI_CONTEXT}} with the normalized sampled context and {{INJECTION_NOTE}} with the injected note; no other prompt text is redacted.",
                "conditions": templates,
                "executed_source_sha256": sha256_file(
                    paper_root / RUNS_REL / HOSTED_RUN / "executed_run_hosted_round3_cc0.py"
                ),
            }
        )
    finally:
        try:
            sys.path.remove(str(src_dir))
        except ValueError:
            pass


def _project_stage_ledger(paper_root: Path) -> bytes:
    source = paper_root / RUNS_REL / REPLAY_RUN / "stage_ledger.jsonl"
    projected = []
    for row in _load_jsonl(source):
        stages = row.get("stages") or {}
        stage_projection: dict[str, Any] = {}
        for stage_name in (
            "source_integrity",
            "json_parse",
            "schema_validation",
            "canonicalization",
            "nonempty_extraction_gate",
            "ontology_load",
            "pellet",
            "inference_ledger",
            "serialization",
        ):
            stage = stages.get(stage_name) or {}
            kept = {
                key: stage[key]
                for key in (
                    "status",
                    "extraction_status",
                    "reasoning_status",
                    "entity_count",
                    "relation_count",
                    "inferred_relation_count",
                    "relations_with_proofs",
                )
                if key in stage and not isinstance(stage[key], (list, dict))
            }
            stage_projection[stage_name] = kept
        case_id = str(row["case_id"])
        projected.append(
            {
                "public_case_id": sha256_bytes(case_id.encode("utf-8"))[:20],
                "condition": row.get("condition"),
                "repeat": row.get("repeat"),
                "context_sha256": row.get("context_sha256"),
                "raw_response_sha256": row.get("raw_response_sha256"),
                "raw_text_sha256": row.get("raw_text_sha256"),
                "pipeline_status": row.get("pipeline_status"),
                "hosted_call_attempts": row.get("hosted_call_attempts"),
                "replay_client_calls": row.get("replay_client_calls"),
                "runtime_seconds": row.get("runtime_seconds"),
                "stages": stage_projection,
            }
        )
    if not projected:
        raise ValueError(f"expected replay ledger rows, found {len(projected)}")
    return stable_jsonl(projected)


def build_s2_payload(paper_root: Path) -> list[PayloadFile]:
    files: list[PayloadFile] = []
    files.append(generated_payload("prompts.json", _prompt_payload(paper_root)))

    hosted_rel = RUNS_REL / HOSTED_RUN
    for name in (
        "executed_run_hosted_round3_cc0.py",
        "run_config.json",
        "metrics.json",
        "gate_G1_schedule_integrity.json",
        "credential_scan_pre_execution.json",
        "credential_scan_post_execution.json",
        "pricing_snapshot.json",
        "pre_call_manifest.json",
        "manifest.json",
        "paired_displacement.json",
    ):
        files.append(
            payload_from_file(paper_root, hosted_rel / name, f"hosted_run/{name}")
        )

    replay_rel = RUNS_REL / REPLAY_RUN
    for name in (
        "README.md",
        "metrics.json",
        "gate_G6_replay_integrity.json",
        "gate_G6_standard_result_integrity.json",
        "manifest.json",
    ):
        files.append(
            payload_from_file(paper_root, replay_rel / name, f"offline_replay/{name}")
        )
    files.append(
        generated_payload("offline_replay/stage_ledger_public.jsonl", _project_stage_ledger(paper_root))
    )

    cluster_rel = RUNS_REL / CLUSTER_RUN
    files.append(_projected_cluster_readme(paper_root))
    for name in (
        "metrics.json",
        "gate_G6_clustered_reanalysis.json",
        "gate_G8_claim_permission.json",
        "source_integrity.json",
        "manifest.json",
        "manifest_verification.json",
    ):
        files.append(
            payload_from_file(paper_root, cluster_rel / name, f"cluster_analysis/{name}")
        )
    return _finalize_payload("S2-hosted-audit-public-v2", files, _hosted_readme())


def write_archive(path: Path, files: Iterable[PayloadFile]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=ARCHIVE_MTIME, compresslevel=9) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for item in sorted(files, key=lambda value: value.archive_path):
                    info = tarfile.TarInfo(item.archive_path)
                    info.size = len(item.data)
                    info.mtime = ARCHIVE_MTIME
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mode = item.mode
                    info.pax_headers = {}
                    archive.addfile(info, io.BytesIO(item.data))
    return sha256_file(path)


def verify_archive(path: Path) -> list[str]:
    failures: list[str] = []
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if names != sorted(names):
            failures.append("members_not_sorted")
        if len(names) != len(set(names)):
            failures.append("duplicate_members")
        for member in members:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or not member.isfile():
                failures.append(f"unsafe_member:{member.name}")
            if member.mtime != ARCHIVE_MTIME or member.uid != 0 or member.gid != 0:
                failures.append(f"noncanonical_metadata:{member.name}")
        try:
            manifest_stream = archive.extractfile("MANIFEST.sha256")
        except KeyError:
            manifest_stream = None
        if manifest_stream is None:
            failures.append("missing_internal_manifest")
            return failures
        for line in manifest_stream.read().decode("utf-8").splitlines():
            expected, name = line.split("  ", 1)
            try:
                stream = archive.extractfile(name)
            except KeyError:
                stream = None
            if stream is None:
                failures.append(f"missing_manifest_member:{name}")
            elif sha256_bytes(stream.read()) != expected:
                failures.append(f"internal_hash_mismatch:{name}")
    return failures


def scan_archive(path: Path) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            lowered = member.name.casefold()
            for forbidden in FORBIDDEN_MEMBER_PARTS:
                if forbidden.casefold() in lowered:
                    findings.append(
                        {"member": member.name, "rule": f"forbidden_member:{forbidden}"}
                    )
            stream = archive.extractfile(member)
            if stream is None:
                continue
            data = stream.read()
            users_prefix = b"/" + b"Users/"
            home_prefix = b"/" + b"home/"
            var_folders_prefix = b"/" + b"var/folders/"
            private_var_prefix = b"/" + b"private/var/"
            repository_marker = b"DO_A_PAPER_" + b"Jul_07_Skill_Update"
            if any(
                marker in data
                for marker in (
                    users_prefix,
                    home_prefix,
                    var_folders_prefix,
                    private_var_prefix,
                    repository_marker,
                )
            ):
                findings.append({"member": member.name, "rule": "personal_absolute_path"})
            for rule, pattern in SECRET_PATTERNS.items():
                if pattern.search(data):
                    findings.append({"member": member.name, "rule": rule})
    return {
        "archive": path.name,
        "finding_count": len(findings),
        "findings": findings,
        "status": "PASS" if not findings else "FAIL",
    }


def _write_release_files(output_dir: Path, archive_hashes: Mapping[str, str]) -> None:
    checksums = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(archive_hashes.items())
    )
    (output_dir / "SHA256SUMS").write_text(checksums, encoding="utf-8", newline="\n")
    verifier = output_dir / "verify_supplements.py"
    verifier.write_bytes(OUTER_VERIFIER)
    verifier.chmod(EXECUTABLE_MODE)


def build_release(paper_root: Path, output_dir: Path) -> dict[str, Any]:
    paper_root = paper_root.resolve()
    output_dir = output_dir.resolve()
    if not (paper_root / "src" / "reasoning.py").is_file():
        raise FileNotFoundError(f"not an AutoMalKG paper root: {paper_root}")

    payloads = {
        S1_FILENAME: build_s1_payload(paper_root),
        S2_FILENAME: build_s2_payload(paper_root),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    scans: dict[str, Any] = {}
    for name, files in payloads.items():
        archive_path = output_dir / name
        hashes[name] = write_archive(archive_path, files)
        verification = verify_archive(archive_path)
        if verification:
            raise RuntimeError(f"archive verification failed for {name}: {verification}")
        scans[name] = scan_archive(archive_path)
        if scans[name]["status"] != "PASS":
            raise RuntimeError(f"archive safety scan failed for {name}: {scans[name]['findings']}")
    _write_release_files(output_dir, hashes)
    report = {
        "schema_version": "1.0",
        "archives": {
            name: {
                "bytes": (output_dir / name).stat().st_size,
                "sha256": digest,
                "internal_verification": "PASS",
                "safety_scan": scans[name],
            }
            for name, digest in sorted(hashes.items())
        },
        "deterministic_archive_contract": {
            "sorted_members": True,
            "gzip_mtime": ARCHIVE_MTIME,
            "tar_mtime": ARCHIVE_MTIME,
            "uid": 0,
            "gid": 0,
        },
        "excluded_content": sorted(FORBIDDEN_MEMBER_PARTS),
        "verification_command": "python3 verify_supplements.py --directory .",
    }
    (output_dir / "build_report.json").write_bytes(stable_json(report))
    return report


def reproducibility_check(paper_root: Path, output_dir: Path) -> dict[str, str]:
    """Build a second release and require byte-identical archives."""

    with tempfile.TemporaryDirectory(prefix="automalkg-supplement-rebuild-") as temporary:
        second = Path(temporary)
        first_report = build_release(paper_root, output_dir)
        second_report = build_release(paper_root, second)
        matches: dict[str, str] = {}
        for name in (S1_FILENAME, S2_FILENAME):
            first_hash = first_report["archives"][name]["sha256"]
            second_hash = second_report["archives"][name]["sha256"]
            if first_hash != second_hash or (output_dir / name).read_bytes() != (second / name).read_bytes():
                raise RuntimeError(f"non-deterministic archive: {name}")
            matches[name] = first_hash
        report_path = output_dir / "build_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["second_build_byte_identity"] = {
            "status": "PASS",
            "archives": matches,
        }
        report_path.write_bytes(stable_json(report))
        return matches


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, default=script_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_root / "8_Project_Management" / "supplementary_archives",
    )
    parser.add_argument(
        "--check-reproducible",
        action="store_true",
        help="build twice in independent directories and require byte identity",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_reproducible:
        hashes = reproducibility_check(args.paper_root, args.output_dir)
    else:
        report = build_release(args.paper_root, args.output_dir)
        hashes = {name: row["sha256"] for name, row in report["archives"].items()}
    for name, digest in sorted(hashes.items()):
        print(f"{digest}  {name}")
    print("PASS: anonymous supplement build and verification completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

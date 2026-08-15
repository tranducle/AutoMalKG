from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import build_anonymous_supplements as supplements


def test_sanitize_text_removes_personal_paths(tmp_path: Path):
    paper_root = tmp_path / "repo" / "Papers" / "paper"
    paper_root.mkdir(parents=True)
    personal_path = "/" + "Users/alice/private.txt"
    temporary_path = "/var/" + "folders/ab/tmp"
    text = f"{paper_root}/src/run.py {personal_path} {temporary_path}"

    cleaned, changed = supplements.sanitize_text(text, paper_root=paper_root)

    assert changed
    assert ("/" + "Users/") not in cleaned
    assert ("/var/" + "folders/") not in cleaned
    assert "./src/run.py" in cleaned


def test_packaged_builder_and_replay_runner_preserve_executable_source_bytes():
    paper_root = Path(supplements.__file__).resolve().parents[1]

    for relative in (
        Path("src/build_anonymous_supplements.py"),
        Path("src/run_stage_reachability_ablation.py"),
    ):
        payload = supplements.payload_from_file(
            paper_root, relative, relative.as_posix()
        )
        assert payload.transformed is False
        assert payload.data == (paper_root / relative).read_bytes()


def test_core_readme_uses_public_fixture_only_and_minimal_semantic_replays():
    readme = supplements._core_readme().decode("utf-8")

    assert "--fixtures-only" in readme
    assert "--consistency-cases-per-class 1" in readme
    assert "--swrl-controls-per-class 1" in readme
    assert "29 synthetic contract observations" in readme
    assert "requirements-reproduction.txt" in readme
    assert b"pytest==" in supplements.REPRODUCTION_REQUIREMENTS
    assert b"scipy==" in supplements.REPRODUCTION_REQUIREMENTS


def test_public_stage_ledger_projection_excludes_text_fields(tmp_path: Path):
    paper_root = tmp_path / "paper"
    ledger = paper_root / supplements.RUNS_REL / supplements.REPLAY_RUN / "stage_ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    row = {
        "case_id": "named-group-secret-case",
        "condition": "typed_evidence",
        "repeat": 1,
        "context_sha256": "a" * 64,
        "raw_response_sha256": "b" * 64,
        "raw_text_sha256": "c" * 64,
        "pipeline_status": "success",
        "hosted_call_attempts": 0,
        "replay_client_calls": 1,
        "runtime_seconds": 0.1,
        "diagnostics": {"extraction": ["SECRET ENTITY TEXT"]},
        "source_path": "data/private/report.json",
        "stages": {
            "canonicalization": {"status": "passed", "entity_count": 2},
            "inference_ledger": {
                "status": "passed",
                "inferred_relation_count": 1,
                "inferred_relations": [{"subject": "SECRET ENTITY TEXT"}],
            },
        },
    }
    ledger.write_text((json.dumps(row) + "\n") * 126, encoding="utf-8")

    projected = supplements._project_stage_ledger(paper_root)

    assert b"SECRET ENTITY TEXT" not in projected
    assert b"named-group-secret-case" not in projected
    assert b"source_path" not in projected
    first = json.loads(projected.splitlines()[0])
    assert len(first["public_case_id"]) == 20
    assert first["stages"]["canonicalization"]["entity_count"] == 2


def test_deterministic_tar_metadata_and_internal_manifest(tmp_path: Path):
    payload = supplements._finalize_payload(
        "test-package",
        [supplements.generated_payload("nested/value.txt", b"value\n")],
        b"readme\n",
    )
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    first_hash = supplements.write_archive(first, payload)
    second_hash = supplements.write_archive(second, payload)

    assert first_hash == second_hash
    assert first.read_bytes() == second.read_bytes()
    assert supplements.verify_archive(first) == []
    with tarfile.open(first, "r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == sorted(member.name for member in members)
        assert all(member.mtime == 0 for member in members)
        assert all(member.uid == 0 and member.gid == 0 for member in members)


def test_scan_archive_rejects_forbidden_hosted_artifact(tmp_path: Path):
    archive = tmp_path / "unsafe.tar.gz"
    supplements.write_archive(
        archive,
        [supplements.generated_payload("hosted_run/raw_responses.jsonl", b"{}\n")],
    )

    report = supplements.scan_archive(archive)

    assert report["status"] == "FAIL"
    assert any("raw_responses.jsonl" in row["rule"] for row in report["findings"])


def test_scan_archive_rejects_unix_home_and_temporary_paths(tmp_path: Path):
    archive = tmp_path / "unsafe-paths.tar.gz"
    unsafe = (
        "/" + "home/alice/work\n"
        + "/" + "var/folders/ab/tmp\n"
        + "/" + "private/var/tmp/work\n"
    ).encode("utf-8")
    supplements.write_archive(
        archive,
        [supplements.generated_payload("notes.txt", unsafe)],
    )

    report = supplements.scan_archive(archive)

    assert report["status"] == "FAIL"
    assert sum(row["rule"] == "personal_absolute_path" for row in report["findings"]) == 1


def test_internal_verifier_payload_has_no_external_dependencies():
    assert b"import requests" not in supplements.INTERNAL_VERIFIER
    assert b"MANIFEST.sha256" in supplements.INTERNAL_VERIFIER

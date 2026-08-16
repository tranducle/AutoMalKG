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


def test_hosted_readme_scopes_the_ctihal_statement_to_the_superseded_reanalysis():
    """The exclusion paragraph must not read as though the CC0 corpus were withheld.

    Its earlier wording listed "CTI-HAL text, source excerpts" among the
    exclusions and then said licensing was not established, which contradicted
    the manuscript's statement that all 16 evaluation contexts are
    author-authored and CC0-dedicated.
    """

    readme = supplements._hosted_readme().decode("utf-8")

    assert "The evaluation corpus is not withheld" in readme
    assert "benchmark/round3_cc0/" in readme
    assert "CC0-1.0" in readme
    # The CTI-HAL statement survives, but bound to the superseded reanalysis.
    ctihal = readme[readme.index("CTI-HAL, which the") :]
    assert "superseded stress run projected in `cluster_analysis/`" in ctihal
    assert "not on the reported round-3 run" in ctihal
    # Every top-level member of the package is indexed.
    for entry in (
        "`prompts.json`",
        "`hosted_run/`",
        "`offline_replay/`",
        "`cluster_analysis/`",
        "`MANIFEST.sha256`",
        "`PACKAGE_METADATA.json`",
    ):
        assert entry in readme


def test_projected_cluster_readme_carries_the_supersession_note(tmp_path: Path):
    paper_root = tmp_path / "paper"
    source = paper_root / supplements.RUNS_REL / supplements.CLUSTER_RUN / "README.md"
    source.parent.mkdir(parents=True)
    body = "# Hosted Clustered Reanalysis\n\nBody line.\n"
    source.write_text(body, encoding="utf-8")

    payload = supplements._projected_cluster_readme(paper_root)
    text = payload.data.decode("utf-8")

    assert text.startswith("> **Superseded")
    assert "hosted_stress_20260809T211342Z" in text
    assert "hosted_round3_cc0_20260810T004042Z" in text
    assert text.endswith(body)
    # The frozen run directory is covered by its own manifest and stays untouched.
    assert source.read_text(encoding="utf-8") == body
    assert payload.source_sha256 == supplements.sha256_bytes(body.encode("utf-8"))

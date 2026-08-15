"""Regression tests for hosted-call provenance and legacy-result safeguards."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PAPER_ROOT = Path(__file__).resolve().parents[1]
SRC = PAPER_ROOT / "src"
sys.path.insert(0, str(SRC))

import baselines  # noqa: E402
import llm_client  # noqa: E402
import run_all  # noqa: E402
import run_real_experiments  # noqa: E402
from contracts import ExtractionResult, ExtractionStatus  # noqa: E402


class _FakeCompletions:
    def __init__(self, outcome):
        self.outcome = outcome

    async def create(self, **kwargs):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        self.outcome.request = kwargs
        return self.outcome


class _FakeClient:
    def __init__(self, outcome):
        self.chat = SimpleNamespace(completions=_FakeCompletions(outcome))


class _FakeResponse:
    def __init__(self):
        self.id = "req_test_123"
        self.model = "provider/resolved-model-20260809"
        self.created = 1786320000
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content='{"entities": []}'))
        ]
        self.usage = SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            prompt_tokens_details=SimpleNamespace(cached_tokens=3),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
        )
        self.request = None

    def model_dump_json(self, **kwargs):
        return json.dumps(
            {
                "id": self.id,
                "model": self.model,
                "content": self.choices[0].message.content,
            },
            sort_keys=True,
        )


class _ProviderFailure(RuntimeError):
    status_code = 429
    code = "rate_limit"


def test_hosted_response_records_actual_metadata_and_sanitized_retries():
    secret = "<REDACTED_TOKEN>"
    response = _FakeResponse()
    client = llm_client.MultiModelClient(
        providers=[
            {
                "name": "first",
                "client": _FakeClient(_ProviderFailure(secret)),
                "model": "provider/requested-a",
                "priority": 1,
            },
            {
                "name": "second",
                "client": _FakeClient(response),
                "model": "provider/requested-b",
                "priority": 2,
            },
        ]
    )

    result = asyncio.run(
        client.generate_response(
            "system", "user", max_completion_tokens=2048
        )
    )

    assert result.text == '{"entities": []}'
    assert result.provider == "second"
    assert result.requested_model == "provider/requested-b"
    assert result.resolved_model == "provider/resolved-model-20260809"
    assert result.request_id == "req_test_123"
    assert result.token_usage == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "cached_tokens": 3,
        "reasoning_tokens": 2,
    }
    assert len(result.raw_response_sha256) == 64
    assert [attempt.outcome for attempt in result.retry_history] == [
        "failed",
        "success",
    ]
    serialized = json.dumps(result.to_dict())
    assert secret not in serialized
    assert response.request["model"] == "provider/requested-b"
    assert response.request["max_completion_tokens"] == 2048
    assert client.last_response is result


def test_hosted_client_rejects_nonpositive_completion_cap_before_call():
    response = _FakeResponse()
    client = llm_client.MultiModelClient(
        providers=[
            {
                "name": "provider",
                "client": _FakeClient(response),
                "model": "provider/model",
                "priority": 1,
            }
        ]
    )
    with pytest.raises(ValueError, match="must be positive"):
        asyncio.run(
            client.generate_response(
                "system", "user", max_completion_tokens=0
            )
        )
    assert response.request is None


def test_all_provider_failures_raise_typed_sanitized_error():
    secret = "<REDACTED_TOKEN>"
    client = llm_client.MultiModelClient(
        providers=[
            {
                "name": "only",
                "client": _FakeClient(_ProviderFailure(secret)),
                "model": "provider/requested",
                "priority": 1,
            }
        ]
    )
    with pytest.raises(llm_client.AllProvidersFailedError) as caught:
        asyncio.run(client.generate_response("system", "user"))

    payload = caught.value.to_dict()
    assert payload["retry_history"][0]["status_code"] == 429
    assert payload["retry_history"][0]["error_code"] == "rate_limit"
    assert secret not in json.dumps(payload)
    assert client.last_response is None


def test_empty_provider_set_fails_typed():
    client = llm_client.MultiModelClient(providers=[])
    with pytest.raises(llm_client.AllProvidersFailedError) as caught:
        asyncio.run(client.generate_response("system", "user"))
    assert caught.value.retry_history == ()


def test_openrouter_rejects_short_model_alias(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_FILTER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEYS", "dummy-not-used")
    monkeypatch.setenv("OPENROUTER_MODEL", "qwen3.7-plus")
    with pytest.raises(ValueError, match="full provider/model slug"):
        llm_client.MultiModelClient(aimodel_path=str(PAPER_ROOT / "missing.aimodel"))


def test_openrouter_accepts_singular_key_without_retaining_it_in_metadata(monkeypatch):
    secret = "<REDACTED_TOKEN>"
    monkeypatch.setenv("LLM_PROVIDER_FILTER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEYS", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    monkeypatch.setenv("OPENROUTER_MODEL", "qwen/qwen3.7-plus")
    monkeypatch.setattr(
        llm_client,
        "AsyncOpenAI",
        lambda **_kwargs: SimpleNamespace(chat=SimpleNamespace()),
    )

    client = llm_client.MultiModelClient(
        aimodel_path=str(PAPER_ROOT / "missing.aimodel")
    )

    assert len(client.providers) == 1
    assert client.providers[0]["name"] == "openrouter_key_1"
    assert client.providers[0]["model"] == "qwen/qwen3.7-plus"
    assert secret not in json.dumps(
        {
            "name": client.providers[0]["name"],
            "model": client.providers[0]["model"],
        }
    )


def test_legacy_export_is_deterministic_and_explicitly_non_evidence(tmp_path):
    source = tmp_path / "legacy.json"
    source.write_text('{"metric": 0.9}\n', encoding="utf-8")
    output = tmp_path / "export.json"

    result_path = run_all.export_archive(
        "exp1", source_path=source, output_path=output
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert payload["artifact_type"] == "legacy_archive_export"
    assert payload["evidence_status"] == run_all.NON_EVIDENCE_STATUS
    assert "not a rerun" in payload["warning"]
    assert payload["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert payload["archived_payload"] == {"metric": 0.9}


def test_legacy_export_fails_on_missing_or_overwrite(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(run_all.ArchiveExportError, match="missing"):
        run_all.export_archive("exp2", source_path=missing, output_path=tmp_path / "out")

    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(run_all.ArchiveExportError, match="must not overwrite"):
        run_all.export_archive("exp2", source_path=source, output_path=source)


def test_legacy_entrypoints_contain_no_archived_metric_literals():
    forbidden = ("0.929", "0.0034", "342", "40.69", "ANOVA_p_value")
    for filename in (
        "exp1_extraction.py",
        "exp2_ablation.py",
        "exp3_robustness.py",
        "exp4_scalability.py",
        "exp5_case_study.py",
    ):
        source = (SRC / filename).read_text(encoding="utf-8")
        assert all(value not in source for value in forbidden)
        assert "does not run Experiment" in source


def test_generic_ner_cannot_be_relabelled_as_cyber_roberta():
    models = baselines.BaselineModels()
    with pytest.raises(ValueError, match="generic NER checkpoint"):
        models.load_roberta("dslim/bert-base-NER")
    with pytest.raises(baselines.BaselineUnavailableError, match="not been loaded"):
        models.extract_roberta("APT29 uses malware")


def test_requirements_exclude_stdlib_asyncio_and_bound_direct_dependencies():
    requirements = (PAPER_ROOT / "requirements.txt").read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any(line.split("=", 1)[0].lower() == "asyncio" for line in lines)
    assert all(any(operator in line for operator in ("==", ">=", "~=", "<")) for line in lines)
    assert "Java" in requirements


def test_real_runner_uses_synthetic_cve_fixture_and_records_failed_attempt(
    tmp_path, monkeypatch
):
    class FakeClient:
        last_response = None

    class FakeExtractor:
        def __init__(self, client):
            self.client = client

        async def extract_from_chunk(self, text):
            return ExtractionResult(
                ExtractionStatus.CLIENT_ERROR,
                diagnostics=["controlled hosted failure"],
            )

    artifact = tmp_path / "attempt.json"
    monkeypatch.setattr(run_real_experiments, "MultiModelClient", FakeClient)
    monkeypatch.setattr(run_real_experiments, "MalOntExtractor", FakeExtractor)
    monkeypatch.setattr(run_real_experiments, "BENCHMARK_PATH", artifact)

    with pytest.raises(RuntimeError, match="failed closed"):
        asyncio.run(run_real_experiments.main())

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert "CVE-2020-10148" in run_real_experiments.SAMPLE_TEXT
    assert "SolarWinds_Orion" not in run_real_experiments.SAMPLE_TEXT
    assert payload["run_status"] == "failed_extraction"
    assert payload["evidence_status"] == "smoke_test_only_not_accuracy_evidence"

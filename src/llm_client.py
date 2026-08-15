"""OpenAI-compatible hosted inference with evidence-bearing call metadata.

The client never serializes credentials, request headers, endpoint URLs, or raw
provider errors.  Successful responses carry enough metadata to identify and
audit the actual hosted call; exhausted fallback chains raise a typed error.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from openai import AsyncOpenAI


logger = logging.getLogger("MultiModelClient")
_SAFE_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_aimodel_env(filepath: str) -> None:
    """Load missing shell-style exports without overriding the process env."""

    path = Path(filepath)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("export "):
                continue
            key_value = line[7:].split("=", 1)
            if len(key_value) != 2:
                continue
            key, value = key_value
            key = key.strip()
            if not _SAFE_ENV_NAME.fullmatch(key):
                continue
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def resolve_aimodel_path(explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        return explicit_path

    env_path = os.getenv("AIMODEL_PATH")
    if env_path:
        return env_path

    search_roots = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    checked = set()
    for root in search_roots:
        for candidate_dir in (root, *root.parents):
            if candidate_dir in checked:
                continue
            checked.add(candidate_dir)
            candidate = candidate_dir / ".aimodel"
            if candidate.exists():
                return str(candidate)
    return ".aimodel"


def provider_enabled(provider_filter: str, family_name: str) -> bool:
    return not provider_filter or provider_filter == "all" or provider_filter == family_name


def parse_openrouter_key_ordinal(keys_count: int) -> int:
    raw = os.getenv("OPENROUTER_KEY_ORDINAL", "1").strip()
    try:
        ordinal = int(raw)
    except ValueError:
        ordinal = 1
    if keys_count <= 0:
        return 0
    return max(1, min(ordinal, keys_count)) - 1


def _safe_scalar(value: Any) -> Optional[Any]:
    """Return a short non-secret scalar suitable for failure metadata."""

    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    if len(text) > 128 or not re.fullmatch(r"[A-Za-z0-9_.:/-]+", text):
        return None
    return text


@dataclass(frozen=True)
class ProviderAttempt:
    """Sanitized metadata for one provider attempt in a fallback chain."""

    provider: str
    requested_model: str
    timestamp_utc: str
    latency_seconds: float
    outcome: str
    error_type: Optional[str] = None
    status_code: Optional[int] = None
    error_code: Optional[Any] = None

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "requested_model": self.requested_model,
            "timestamp_utc": self.timestamp_utc,
            "latency_seconds": self.latency_seconds,
            "outcome": self.outcome,
            "error_type": self.error_type,
            "status_code": self.status_code,
            "error_code": self.error_code,
        }


@dataclass(frozen=True)
class LLMResponse:
    """Response text paired with reproducibility and provenance metadata."""

    text: str
    provider: str
    requested_model: str
    resolved_model: str
    request_id: Optional[str]
    timestamp_utc: str
    latency_seconds: float
    token_usage: Mapping[str, int]
    raw_response_sha256: str
    retry_history: Tuple[ProviderAttempt, ...]

    def to_dict(self, *, include_text: bool = True) -> dict:
        result = {
            "provider": self.provider,
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "request_id": self.request_id,
            "timestamp_utc": self.timestamp_utc,
            "latency_seconds": self.latency_seconds,
            "token_usage": dict(self.token_usage),
            "raw_response_sha256": self.raw_response_sha256,
            "retry_history": [attempt.to_dict() for attempt in self.retry_history],
        }
        if include_text:
            result["text"] = self.text
        return result


class AllProvidersFailedError(RuntimeError):
    """Raised when no configured provider returns a usable response."""

    def __init__(self, retry_history: Iterable[ProviderAttempt]) -> None:
        self.retry_history = tuple(retry_history)
        if self.retry_history:
            message = f"all {len(self.retry_history)} configured provider attempts failed"
        else:
            message = "no configured LLM provider is available"
        super().__init__(message)

    def to_dict(self) -> dict:
        return {
            "error_type": type(self).__name__,
            "message": str(self),
            "retry_history": [attempt.to_dict() for attempt in self.retry_history],
        }


def _token_usage(response: Any) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    values = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, name, None)
        if isinstance(value, int):
            values[name] = value
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    for output_name, details, attr in (
        ("cached_tokens", prompt_details, "cached_tokens"),
        ("reasoning_tokens", completion_details, "reasoning_tokens"),
    ):
        value = getattr(details, attr, None) if details is not None else None
        if isinstance(value, int):
            values[output_name] = value
    return values


def _raw_response_sha256(response: Any) -> str:
    """Hash the provider response without retaining its raw content."""

    try:
        raw = response.model_dump_json(exclude_none=False).encode("utf-8")
    except (AttributeError, TypeError, ValueError):
        selected = {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "created": getattr(response, "created", None),
            "usage": _token_usage(response),
            "text": getattr(
                getattr(getattr(response, "choices", [None])[0], "message", None),
                "content",
                None,
            ),
        }
        raw = json.dumps(selected, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class MultiModelClient:
    """Try configured hosted providers in priority order, preserving provenance."""

    def __init__(
        self,
        aimodel_path: Optional[str] = None,
        *,
        providers: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        self.last_response: Optional[LLMResponse] = None
        if providers is not None:
            self.providers = sorted(
                [dict(provider) for provider in providers],
                key=lambda provider: provider.get("priority", 0),
            )
            return

        load_aimodel_env(resolve_aimodel_path(aimodel_path))
        provider_filter = os.getenv("LLM_PROVIDER_FILTER", "").strip().lower()
        configured = []

        alibaba_key = os.getenv("ALIBABA_API_KEY")
        if alibaba_key and provider_enabled(provider_filter, "alibaba"):
            alibaba_base = os.getenv("ALIBABA_BASE_URL", "").replace(
                "/chat/completions", ""
            )
            for name, model, priority in (
                (
                    "alibaba_primary",
                    os.getenv("ALIBABA_MODEL_PRIMARY", "qwen3.7-plus"),
                    1,
                ),
                (
                    "alibaba_fallback",
                    os.getenv("ALIBABA_MODEL_FALLBACK", "qwen3.6-plus"),
                    2,
                ),
            ):
                configured.append(
                    {
                        "name": name,
                        "client": AsyncOpenAI(api_key=alibaba_key, base_url=alibaba_base),
                        "model": model,
                        "priority": priority,
                    }
                )

        openrouter_keys = [
            key.strip()
            for key in os.getenv("OPENROUTER_API_KEYS", "").split(",")
            if key.strip()
        ]
        single_openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if single_openrouter_key and single_openrouter_key not in openrouter_keys:
            openrouter_keys.append(single_openrouter_key)
        if openrouter_keys and provider_enabled(provider_filter, "openrouter"):
            openrouter_base = os.getenv("OPENROUTER_BASE_URL", "").replace(
                "/chat/completions", ""
            )
            # Use the provider's full immutable routing slug, never a generic alias.
            openrouter_model = os.getenv(
                "OPENROUTER_MODEL", "qwen/qwen3.7-plus"
            ).strip()
            if not openrouter_model or "/" not in openrouter_model:
                raise ValueError(
                    "OPENROUTER_MODEL must be a full provider/model slug, not an alias"
                )
            preferred = parse_openrouter_key_ordinal(len(openrouter_keys))
            indices = [preferred] + [
                index for index in range(len(openrouter_keys)) if index != preferred
            ]
            for offset, key_index in enumerate(indices):
                configured.append(
                    {
                        "name": f"openrouter_key_{key_index + 1}",
                        "client": AsyncOpenAI(
                            api_key=openrouter_keys[key_index], base_url=openrouter_base
                        ),
                        "model": openrouter_model,
                        "priority": 30 + offset,
                    }
                )

        inferstack_key = os.getenv("INFERSTACK_API_KEY")
        if inferstack_key and provider_enabled(provider_filter, "inferstack"):
            configured.append(
                {
                    "name": "inferstack",
                    "client": AsyncOpenAI(
                        api_key=inferstack_key,
                        base_url=os.getenv("INFERSTACK_BASE_URL", "").replace(
                            "/chat/completions", ""
                        ),
                    ),
                    "model": os.getenv("INFERSTACK_MODEL", "gpt-5.4"),
                    "priority": 4,
                }
            )

        opencode_key = os.getenv("OPENCODE_API_KEY")
        if opencode_key and provider_enabled(provider_filter, "opencode"):
            configured.append(
                {
                    "name": "opencode",
                    "client": AsyncOpenAI(
                        api_key=opencode_key,
                        base_url=os.getenv("OPENCODE_BASE_URL", "").replace(
                            "/chat/completions", ""
                        ),
                    ),
                    "model": os.getenv(
                        "OPENCODE_MODEL", "deepseek-v4-pro-opencode"
                    ),
                    "priority": 5,
                }
            )

        self.providers = sorted(configured, key=lambda provider: provider["priority"])
        if not self.providers:
            logger.warning("No LLM providers loaded; hosted inference will fail closed")

    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_completion_tokens: int | None = None,
    ) -> LLMResponse:
        """Return response text plus actual call metadata, or fail typed."""

        if max_completion_tokens is not None and max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be positive when provided")

        history = []
        self.last_response = None
        for provider in self.providers:
            started_at = _utc_now()
            started = time.perf_counter()
            provider_name = str(provider["name"])
            requested_model = str(provider["model"])
            try:
                request_options: dict[str, Any] = {
                    "model": requested_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                }
                if max_completion_tokens is not None:
                    request_options["max_completion_tokens"] = max_completion_tokens
                raw_response = await provider["client"].chat.completions.create(
                    **request_options,
                )
                latency = round(time.perf_counter() - started, 6)
                text = raw_response.choices[0].message.content
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("provider returned empty response content")
                success = ProviderAttempt(
                    provider=provider_name,
                    requested_model=requested_model,
                    timestamp_utc=started_at,
                    latency_seconds=latency,
                    outcome="success",
                )
                history.append(success)
                response = LLMResponse(
                    text=text,
                    provider=provider_name,
                    requested_model=requested_model,
                    resolved_model=str(
                        getattr(raw_response, "model", None) or requested_model
                    ),
                    request_id=_safe_scalar(getattr(raw_response, "id", None)),
                    timestamp_utc=_utc_now(),
                    latency_seconds=latency,
                    token_usage=_token_usage(raw_response),
                    raw_response_sha256=_raw_response_sha256(raw_response),
                    retry_history=tuple(history),
                )
                self.last_response = response
                return response
            except Exception as exc:
                latency = round(time.perf_counter() - started, 6)
                status_code = getattr(exc, "status_code", None)
                if not isinstance(status_code, int):
                    status_code = None
                error_code = _safe_scalar(getattr(exc, "code", None))
                history.append(
                    ProviderAttempt(
                        provider=provider_name,
                        requested_model=requested_model,
                        timestamp_utc=started_at,
                        latency_seconds=latency,
                        outcome="failed",
                        error_type=type(exc).__name__,
                        status_code=status_code,
                        error_code=error_code,
                    )
                )
                logger.warning(
                    "Hosted provider attempt failed: provider=%s error_type=%s status=%s",
                    provider_name,
                    type(exc).__name__,
                    status_code,
                )
        raise AllProvidersFailedError(history)


async def process_chunks_concurrently(
    chunks: Sequence[str], system_prompt: str, max_concurrency: int = 5
) -> list[LLMResponse]:
    """Process chunks in input order; any exhausted call fails the batch closed."""

    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    client = MultiModelClient()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_chunk(chunk: str, index: int) -> tuple[int, LLMResponse]:
        async with semaphore:
            response = await client.generate_response(system_prompt, chunk)
            return index, response

    results = await asyncio.gather(
        *(process_chunk(chunk, index) for index, chunk in enumerate(chunks))
    )
    return [response for _, response in sorted(results, key=lambda item: item[0])]


if __name__ == "__main__":  # pragma: no cover - manual hosted smoke test.
    async def _smoke_test() -> None:
        client = MultiModelClient()
        response = await client.generate_response(
            "You are a helpful assistant.", "Reply with the single word hello."
        )
        print(json.dumps(response.to_dict(include_text=False), indent=2))

    asyncio.run(_smoke_test())

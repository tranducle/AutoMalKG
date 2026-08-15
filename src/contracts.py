"""Typed, JSON-compatible contracts shared by the AutoMalKG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any, Iterable, Mapping, Optional, Sequence


class ExtractionStatus(str, Enum):
    """Terminal states for one extraction or a merged extraction batch."""

    SUCCESS = "success"
    PARTIAL = "partial"
    EMPTY = "empty"
    MALFORMED = "malformed"
    SCHEMA_INVALID = "schema_invalid"
    CLIENT_ERROR = "client_error"


class ReasoningStatus(str, Enum):
    """States emitted while validating and reasoning over an extracted graph."""

    VALIDATED = "validated"
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    VALIDATION_ERROR = "validation_error"
    REASONER_UNAVAILABLE = "reasoner_unavailable"
    REASONER_ERROR = "reasoner_error"
    NOT_RUN = "not_run"


class ExtractionResult(dict):
    """Typed extraction result that remains directly JSON serializable.

    Subclassing ``dict`` preserves compatibility with the original experiment
    scripts, which call ``json.dump(result)`` and ``result.get(...)``.
    """

    def __init__(
        self,
        status: ExtractionStatus,
        *,
        entities: Optional[Sequence[Mapping[str, Any]]] = None,
        relations: Optional[Sequence[Mapping[str, Any]]] = None,
        diagnostics: Optional[Iterable[str]] = None,
        chunk_id: Optional[str] = None,
        chunk_registry: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        super().__init__(
            status=status.value,
            entities=[dict(entity) for entity in (entities or ())],
            relations=[dict(relation) for relation in (relations or ())],
            diagnostics=sorted(set(diagnostics or ())),
        )
        if chunk_id is not None:
            self["chunk_id"] = chunk_id
        if chunk_registry is not None:
            self["chunk_registry"] = {
                key: dict(chunk_registry[key]) for key in sorted(chunk_registry)
            }

    @property
    def status(self) -> ExtractionStatus:
        """Return the status as an enum instead of its serialized string."""

        return ExtractionStatus(self["status"])

    @property
    def succeeded(self) -> bool:
        """Whether the complete requested extraction succeeded."""

        return self.status is ExtractionStatus.SUCCESS


class ReasoningResult(dict):
    """Typed reasoning result with fail-closed truth-value semantics."""

    def __init__(
        self,
        status: ReasoningStatus,
        *,
        diagnostics: Optional[Iterable[str]] = None,
        inferred_relations: Optional[Sequence[Mapping[str, Any]]] = None,
        entity_count: int = 0,
        relation_count: int = 0,
    ) -> None:
        super().__init__(
            status=status.value,
            diagnostics=sorted(set(diagnostics or ())),
            inferred_relations=[
                dict(relation) for relation in (inferred_relations or ())
            ],
            entity_count=int(entity_count),
            relation_count=int(relation_count),
        )

    @property
    def status(self) -> ReasoningStatus:
        """Return the status as an enum instead of its serialized string."""

        return ReasoningStatus(self["status"])

    def __bool__(self) -> bool:
        """Only a completed, consistent reasoning run is truthy."""

        return self.status is ReasoningStatus.CONSISTENT


@dataclass(frozen=True)
class StructuredChunk:
    """A content-addressed text chunk with source offsets and provenance."""

    chunk_id: str
    source_id: str
    document_sha256: str
    index: int
    start_word: int
    end_word: int
    text: str
    text_sha256: str

    @classmethod
    def create(
        cls,
        *,
        text: str,
        source_id: str,
        index: int,
        start_word: int,
        end_word: int,
        document_sha256: Optional[str] = None,
    ) -> "StructuredChunk":
        """Build a stable chunk identity from source, offsets, and content."""

        encoded = text.encode("utf-8")
        text_digest = sha256(encoded).hexdigest()
        document_digest = document_sha256 or sha256(encoded).hexdigest()
        identity = (
            f"{source_id}\0{document_digest}\0{index}\0{start_word}\0"
            f"{end_word}\0{text_digest}"
        )
        chunk_digest = sha256(identity.encode("utf-8")).hexdigest()[:20]
        return cls(
            chunk_id=f"chunk_{chunk_digest}",
            source_id=source_id,
            document_sha256=document_digest,
            index=index,
            start_word=start_word,
            end_word=end_word,
            text=text,
            text_sha256=text_digest,
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""

        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "document_sha256": self.document_sha256,
            "index": self.index,
            "start_word": self.start_word,
            "end_word": self.end_word,
            "text": self.text,
            "text_sha256": self.text_sha256,
        }

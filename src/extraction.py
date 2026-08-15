"""Validated LLM extraction and deterministic cross-chunk canonicalization."""

from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from hashlib import sha256
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

try:  # Support both ``python src/extraction.py`` and package-style imports.
    from .contracts import ExtractionResult, ExtractionStatus, StructuredChunk
    from .llm_client import MultiModelClient
except ImportError:  # pragma: no cover - exercised by script entry points.
    from contracts import ExtractionResult, ExtractionStatus, StructuredChunk
    from llm_client import MultiModelClient


ENTITY_TYPES = frozenset(
    {"ThreatActor", "Malware", "Vulnerability", "Campaign", "Indicator"}
)
RELATION_SIGNATURES = {
    "Uses": (frozenset({"ThreatActor"}), frozenset({"Malware"})),
    "Targets": (frozenset({"Malware"}), frozenset({"Vulnerability"})),
    "AssociatedWith": (frozenset({"Malware"}), frozenset({"Campaign"})),
    "HasIndicator": (frozenset({"Malware"}), frozenset({"Indicator"})),
}
ROOT_FIELDS = frozenset({"entities", "relations"})
RESULT_ENVELOPE_FIELDS = frozenset(
    {"status", "entities", "relations", "diagnostics", "chunk_id", "chunk_registry"}
)
ENTITY_FIELDS = frozenset(
    {"id", "type", "name", "aliases", "provenance", "evidence"}
)
RELATION_FIELDS = frozenset(
    {"subject", "predicate", "object", "provenance", "evidence"}
)
CANONICAL_RELATION_FIELDS = RELATION_FIELDS | frozenset({"id", "key"})


def _canonical_name(name: str) -> str:
    """Normalize an entity label for deterministic identity resolution."""

    normalized = unicodedata.normalize("NFKC", name)
    return " ".join(normalized.split()).casefold()


def _stable_entity_id(entity_type: str, canonical_name: str) -> str:
    """Return a content-addressed entity identifier."""

    digest = sha256(f"{entity_type}\0{canonical_name}".encode("utf-8")).hexdigest()
    return f"ent_{digest[:20]}"


def relation_key(subject: str, predicate: str, obj: str) -> str:
    """Return the unambiguous canonical key for one directed relation."""

    return f"{subject}|{predicate}|{obj}"


def stable_relation_id(subject: str, predicate: str, obj: str) -> str:
    """Return a deterministic identifier for one canonical relation."""

    digest = sha256(relation_key(subject, predicate, obj).encode("utf-8")).hexdigest()
    return f"rel_{digest[:20]}"


def _chunk_metadata(
    *, chunk_id: str, text: str, source_id: str, index: int
) -> dict:
    """Construct resolvable metadata for an unstructured inline chunk."""

    digest = sha256(text.encode("utf-8")).hexdigest()
    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "document_sha256": digest,
        "index": index,
        "start_word": 0,
        "end_word": len(text.split()),
        "text_sha256": digest,
    }


def _coerce_chunk(chunk: Any, index: int) -> Tuple[str, str, dict]:
    """Return ``(chunk_id, text, metadata)`` for supported chunks."""

    if isinstance(chunk, StructuredChunk):
        metadata = chunk.to_dict()
        metadata.pop("text")
        return chunk.chunk_id, chunk.text, metadata
    if isinstance(chunk, Mapping):
        text = chunk.get("text")
        if not isinstance(text, str):
            raise TypeError("structured chunk field 'text' must be a string")
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            digest = sha256(text.encode("utf-8")).hexdigest()[:20]
            chunk_id = f"chunk_{index}_{digest}"
        metadata = _chunk_metadata(
            chunk_id=chunk_id,
            text=text,
            source_id=str(chunk.get("source_id", "inline")),
            index=int(chunk.get("index", index)),
        )
        for field in (
            "document_sha256",
            "start_word",
            "end_word",
            "text_sha256",
        ):
            if field in chunk:
                metadata[field] = chunk[field]
        return chunk_id, text, metadata
    if isinstance(chunk, str):
        digest = sha256(chunk.encode("utf-8")).hexdigest()[:20]
        chunk_id = f"chunk_{index}_{digest}"
        return chunk_id, chunk, _chunk_metadata(
            chunk_id=chunk_id, text=chunk, source_id="inline", index=index
        )
    raise TypeError("chunk must be text, StructuredChunk, or a mapping with text")


def _validated_string_array(
    raw: Mapping[str, Any], field: str, location: str, diagnostics: list
) -> list:
    """Validate and normalize an optional array of non-empty strings."""

    values = raw.get(field, [])
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        diagnostics.append(f"{location}.{field} must be an array of non-empty strings")
        return []
    return sorted(set(value.strip() for value in values))


def _diagnose_unknown_keys(
    raw: Mapping[str, Any], allowed: frozenset[str], location: str, diagnostics: list
) -> None:
    """Append one deterministic, path-scoped diagnostic per unknown key."""

    for key in sorted((key for key in raw if key not in allowed), key=repr):
        diagnostics.append(f"{location} contains unknown key: {key!r}")


def validate_extraction_schema(
    data: Any, *, allow_result_envelope: bool = False
) -> Tuple[list, list, list]:
    """Validate an extraction document without mutating downstream state.

    Returns normalized shallow copies of entities and relations plus a list of
    deterministic diagnostics. Any non-empty diagnostic list makes the schema
    invalid.
    """

    diagnostics = []
    if not isinstance(data, Mapping):
        return [], [], ["root must be a JSON object"]

    root_fields = RESULT_ENVELOPE_FIELDS if allow_result_envelope else ROOT_FIELDS
    _diagnose_unknown_keys(data, root_fields, "root", diagnostics)

    entities_raw = data.get("entities")
    relations_raw = data.get("relations")
    if not isinstance(entities_raw, list):
        diagnostics.append("'entities' must be a list")
        entities_raw = []
    if not isinstance(relations_raw, list):
        diagnostics.append("'relations' must be a list")
        relations_raw = []

    entities = []
    ids = set()
    entity_types = {}
    for index, raw in enumerate(entities_raw):
        if not isinstance(raw, Mapping):
            diagnostics.append(f"entities[{index}] must be an object")
            continue
        _diagnose_unknown_keys(
            raw, ENTITY_FIELDS, f"entities[{index}]", diagnostics
        )
        entity_id = raw.get("id")
        entity_type = raw.get("type")
        name = raw.get("name")
        if not isinstance(entity_id, str) or not entity_id.strip():
            diagnostics.append(f"entities[{index}].id must be a non-empty string")
            continue
        if entity_id in ids:
            diagnostics.append(f"duplicate entity id: {entity_id}")
            continue
        ids.add(entity_id)
        if entity_type not in ENTITY_TYPES:
            diagnostics.append(
                f"entities[{index}].type is not permitted: {entity_type!r}"
            )
            continue
        if not isinstance(name, str) or not name.strip():
            diagnostics.append(f"entities[{index}].name must be a non-empty string")
            continue
        entity_types[entity_id] = entity_type
        location = f"entities[{index}]"
        provenance = _validated_string_array(raw, "provenance", location, diagnostics)
        aliases = _validated_string_array(raw, "aliases", location, diagnostics)
        if "evidence" in raw and not isinstance(raw["evidence"], str):
            diagnostics.append(f"{location}.evidence must be a string")
        record = {"id": entity_id, "type": entity_type, "name": name.strip()}
        if "aliases" in raw:
            record["aliases"] = aliases
        if "provenance" in raw:
            record["provenance"] = provenance
        entities.append(record)

    relations = []
    relation_keys = set()
    for index, raw in enumerate(relations_raw):
        if not isinstance(raw, Mapping):
            diagnostics.append(f"relations[{index}] must be an object")
            continue
        relation_fields = (
            CANONICAL_RELATION_FIELDS if allow_result_envelope else RELATION_FIELDS
        )
        _diagnose_unknown_keys(raw, relation_fields, f"relations[{index}]", diagnostics)
        subject = raw.get("subject")
        predicate = raw.get("predicate")
        obj = raw.get("object")
        if predicate not in RELATION_SIGNATURES:
            diagnostics.append(
                f"relations[{index}].predicate is not permitted: {predicate!r}"
            )
            continue
        if subject not in entity_types:
            diagnostics.append(f"relations[{index}] has dangling subject: {subject!r}")
            continue
        if obj not in entity_types:
            diagnostics.append(f"relations[{index}] has dangling object: {obj!r}")
            continue
        allowed_subjects, allowed_objects = RELATION_SIGNATURES[predicate]
        if entity_types[subject] not in allowed_subjects:
            diagnostics.append(
                f"relations[{index}] subject type {entity_types[subject]} is invalid "
                f"for {predicate}"
            )
            continue
        if entity_types[obj] not in allowed_objects:
            diagnostics.append(
                f"relations[{index}] object type {entity_types[obj]} is invalid "
                f"for {predicate}"
            )
            continue
        key = (subject, predicate, obj)
        if key in relation_keys:
            diagnostics.append(
                f"duplicate relation: {subject}|{predicate}|{obj}"
            )
            continue
        relation_keys.add(key)
        canonical_key = relation_key(subject, predicate, obj)
        canonical_id = stable_relation_id(subject, predicate, obj)
        if "key" in raw and raw["key"] != canonical_key:
            diagnostics.append(
                f"relations[{index}].key does not match its canonical endpoints"
            )
        if "id" in raw and raw["id"] != canonical_id:
            diagnostics.append(
                f"relations[{index}].id does not match its canonical endpoints"
            )
        provenance = _validated_string_array(
            raw, "provenance", f"relations[{index}]", diagnostics
        )
        if "evidence" in raw and not isinstance(raw["evidence"], str):
            diagnostics.append(f"relations[{index}].evidence must be a string")
        record = {
            "id": canonical_id,
            "key": canonical_key,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
        }
        if "provenance" in raw:
            record["provenance"] = provenance
        relations.append(record)

    return entities, relations, sorted(diagnostics)


class MalOntExtractor:
    """Extract MALOnt-compatible facts and merge them deterministically."""

    def __init__(self, client: Optional[Any] = None) -> None:
        self.client = client if client is not None else MultiModelClient()
        self.system_prompt = """
You are an expert Cyber Threat Intelligence (CTI) analyst.
Extract MALOnt entities (ThreatActor, Malware, Vulnerability, Campaign,
Indicator) and relations (Uses, Targets, AssociatedWith, HasIndicator).
Return only JSON with top-level arrays named entities and relations. Every
relation must reference entity IDs declared in the same response.
""".strip()

    async def extract_from_chunk(
        self, chunk: Any, chunk_index: int = 0
    ) -> ExtractionResult:
        """Extract and validate one chunk, preserving its provenance ID."""

        try:
            chunk_id, text, chunk_metadata = _coerce_chunk(chunk, chunk_index)
        except (TypeError, ValueError) as exc:
            return ExtractionResult(
                ExtractionStatus.SCHEMA_INVALID, diagnostics=[str(exc)]
            )
        if not text.strip():
            return ExtractionResult(
                ExtractionStatus.EMPTY,
                diagnostics=["chunk text is empty"],
                chunk_id=chunk_id,
                chunk_registry={chunk_id: chunk_metadata},
            )

        try:
            response = await self.client.generate_response(self.system_prompt, text)
        except Exception as exc:  # Provider failures are data, never empty success.
            return ExtractionResult(
                ExtractionStatus.CLIENT_ERROR,
                diagnostics=[f"LLM client error: {type(exc).__name__}: {exc}"],
                chunk_id=chunk_id,
                chunk_registry={chunk_id: chunk_metadata},
            )
        # Hosted clients may return a typed response carrying provider/model
        # provenance; extraction consumes only its text without coupling to the
        # client's concrete response class.
        if not isinstance(response, str) and hasattr(response, "text"):
            response = response.text
        if not isinstance(response, str) or not response.strip():
            return ExtractionResult(
                ExtractionStatus.EMPTY,
                diagnostics=["LLM response is empty"],
                chunk_id=chunk_id,
                chunk_registry={chunk_id: chunk_metadata},
            )

        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        json_text = fenced.group(1) if fenced else response.strip()
        try:
            data = json.loads(json_text)
        except (json.JSONDecodeError, TypeError) as exc:
            return ExtractionResult(
                ExtractionStatus.MALFORMED,
                diagnostics=[f"invalid JSON: {exc}"],
                chunk_id=chunk_id,
                chunk_registry={chunk_id: chunk_metadata},
            )

        entities, relations, diagnostics = validate_extraction_schema(data)
        if diagnostics:
            return ExtractionResult(
                ExtractionStatus.SCHEMA_INVALID,
                diagnostics=diagnostics,
                chunk_id=chunk_id,
                chunk_registry={chunk_id: chunk_metadata},
            )
        if not entities and not relations:
            return ExtractionResult(
                ExtractionStatus.EMPTY,
                diagnostics=["valid response contains no entities or relations"],
                chunk_id=chunk_id,
                chunk_registry={chunk_id: chunk_metadata},
            )
        for entity in entities:
            entity["provenance"] = sorted(
                set(entity.get("provenance", ())) | {chunk_id}
            )
        for relation in relations:
            relation["provenance"] = sorted(
                set(relation.get("provenance", ())) | {chunk_id}
            )
        return ExtractionResult(
            ExtractionStatus.SUCCESS,
            entities=entities,
            relations=relations,
            chunk_id=chunk_id,
            chunk_registry={chunk_id: chunk_metadata},
        )

    async def process_document_chunks(
        self, chunks: Sequence[Any]
    ) -> ExtractionResult:
        """Extract chunks concurrently and canonically merge successful facts."""

        if not chunks:
            return ExtractionResult(
                ExtractionStatus.EMPTY, diagnostics=["document contains no chunks"]
            )
        tasks = [self.extract_from_chunk(chunk, index) for index, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        return self._canonicalize(results)

    @staticmethod
    def _canonicalize(results: Sequence[ExtractionResult]) -> ExtractionResult:
        """Merge result facts by semantic identity with provenance union."""

        diagnostics = []
        entity_by_key = {}
        id_to_key = {}
        name_types = {}
        local_maps = []
        chunk_registry = {}

        for result_index, result in enumerate(results):
            chunk_id = result.get("chunk_id", f"result_{result_index}")
            for registry_id, metadata in result.get("chunk_registry", {}).items():
                previous = chunk_registry.get(registry_id)
                if previous is not None and previous != metadata:
                    diagnostics.append(
                        f"conflicting chunk metadata for provenance id: {registry_id}"
                    )
                else:
                    chunk_registry[registry_id] = dict(metadata)
            diagnostics.extend(
                f"{chunk_id}: {message}" for message in result.get("diagnostics", [])
            )
            local_map = {}
            local_maps.append(local_map)
            if result.status is not ExtractionStatus.SUCCESS:
                continue
            for entity in result["entities"]:
                canonical = _canonical_name(entity["name"])
                key = (entity["type"], canonical)
                stable_id = _stable_entity_id(*key)
                prior_key = id_to_key.get(stable_id)
                if prior_key is not None and prior_key != key:
                    diagnostics.append(
                        f"stable-id collision: {stable_id} maps to {prior_key} and {key}"
                    )
                    stable_id = f"{stable_id}_{sha256(repr(key).encode()).hexdigest()[:8]}"
                id_to_key[stable_id] = key
                local_map[entity["id"]] = stable_id
                name_types.setdefault(canonical, set()).add(entity["type"])

                record = entity_by_key.setdefault(
                    key,
                    {
                        "id": stable_id,
                        "type": entity["type"],
                        "name": entity["name"],
                        "aliases": set(),
                        "provenance": set(),
                    },
                )
                record["aliases"].add(entity["name"])
                record["aliases"].update(entity.get("aliases", ()))
                record["provenance"].update(entity.get("provenance", ()))
                record["provenance"].add(chunk_id)
                record["name"] = min(
                    record["aliases"], key=lambda value: (value.casefold(), value)
                )

        for canonical, types in sorted(name_types.items()):
            if len(types) > 1:
                diagnostics.append(
                    f"canonical-name type collision: {canonical} -> "
                    f"{','.join(sorted(types))}"
                )

        relation_by_key = {}
        for result, local_map in zip(results, local_maps):
            if result.status is not ExtractionStatus.SUCCESS:
                continue
            chunk_id = result.get("chunk_id", "unknown")
            for relation in result["relations"]:
                key = (
                    local_map[relation["subject"]],
                    relation["predicate"],
                    local_map[relation["object"]],
                )
                record = relation_by_key.setdefault(
                    key,
                    {
                        "subject": key[0],
                        "predicate": key[1],
                        "object": key[2],
                        "provenance": set(),
                    },
                )
                record["provenance"].update(relation.get("provenance", ()))
                record["provenance"].add(chunk_id)

        entities = []
        for record in sorted(entity_by_key.values(), key=lambda item: item["id"]):
            entities.append(
                {
                    "id": record["id"],
                    "type": record["type"],
                    "name": record["name"],
                    "aliases": sorted(
                        record["aliases"], key=lambda value: (value.casefold(), value)
                    ),
                    "provenance": sorted(record["provenance"]),
                }
            )
        relations = []
        for key in sorted(relation_by_key):
            record = relation_by_key[key]
            relations.append(
                {
                    "id": stable_relation_id(*key),
                    "key": relation_key(*key),
                    "subject": record["subject"],
                    "predicate": record["predicate"],
                    "object": record["object"],
                    "provenance": sorted(record["provenance"]),
                }
            )

        statuses = [result.status for result in results]
        successes = sum(status is ExtractionStatus.SUCCESS for status in statuses)
        if successes == len(statuses):
            status = ExtractionStatus.SUCCESS
        elif successes:
            status = ExtractionStatus.PARTIAL
        elif all(item is ExtractionStatus.EMPTY for item in statuses):
            status = ExtractionStatus.EMPTY
        elif len(set(statuses)) == 1:
            status = statuses[0]
        else:
            status = ExtractionStatus.PARTIAL
        return ExtractionResult(
            status,
            entities=entities,
            relations=relations,
            diagnostics=diagnostics,
            chunk_registry=chunk_registry,
        )


if __name__ == "__main__":  # pragma: no cover - manual smoke test.
    async def _smoke_test() -> None:
        extractor = MalOntExtractor()
        result = await extractor.extract_from_chunk(
            "APT29 uses the CozyCar malware.", chunk_index=0
        )
        print(json.dumps(result, indent=2))

    asyncio.run(_smoke_test())

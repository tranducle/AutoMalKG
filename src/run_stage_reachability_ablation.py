#!/usr/bin/env python3
"""Offline validation ablation and OWL/SWRL stage-reachability audit.

This runner never constructs a hosted client. It reuses frozen response text,
compares four increasingly capable local stages, and keeps direct OWL mutation
fixtures separate from end-to-end inputs that the extraction contract can emit.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import itertools
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema
import owlready2 as owl

import reasoning as reasoning_module
from contracts import ExtractionStatus, ReasoningStatus, StructuredChunk
from experiment_support import (
    DEFAULT_RUNS_ROOT,
    PAPER_ROOT,
    build_manifest,
    resolve_java,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from extraction import MalOntExtractor
from reasoning import MalOntReasoner


SEED = 20260809
RUN_ID = "offline_stage_reachability_20260809_v4"
FIXTURE_ONLY_RUN_ID = "offline_stage_reachability_fixture_only_20260809_v6"
EXPECTED_HOSTED_CASES = 126
EXPECTED_SYNTHETIC_FIXTURES = 29
EXPECTED_KNOWN_VIOLATIONS = 26
DEFAULT_SOURCE_RUN = DEFAULT_RUNS_ROOT / "hosted_stress_20260809T211342Z"
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_ROOT / RUN_ID
METHODS = (
    "json_parse_only",
    "strict_json_schema",
    "production_canonicalization",
    "full_automalkg",
)
ENTITY_TYPES = (
    "ThreatActor",
    "Malware",
    "Vulnerability",
    "Campaign",
    "Indicator",
)
PROPERTY_SIGNATURES = {
    "Uses": ("ThreatActor", "Malware"),
    "Targets": ("Entity", "Vulnerability"),
    "AssociatedWith": ("Malware", "Campaign"),
    "HasIndicator": ("Malware", "Indicator"),
}


STRICT_EXTRACTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["entities", "relations"],
    "additionalProperties": False,
    "properties": {
        "entities": {
            "type": "array",
            "uniqueItems": True,
            "items": {
                "type": "object",
                "required": ["id", "type", "name"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "pattern": r".*\S.*"},
                    "type": {"enum": list(ENTITY_TYPES)},
                    "name": {"type": "string", "pattern": r".*\S.*"},
                    "aliases": {
                        "type": "array",
                        "items": {"type": "string", "pattern": r".*\S.*"},
                        "uniqueItems": True,
                    },
                    "provenance": {
                        "type": "array",
                        "items": {"type": "string", "pattern": r".*\S.*"},
                        "uniqueItems": True,
                    },
                    "evidence": {"type": "string"},
                },
            },
        },
        "relations": {
            "type": "array",
            "uniqueItems": True,
            "items": {
                "type": "object",
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False,
                "properties": {
                    "subject": {"type": "string", "pattern": r".*\S.*"},
                    "predicate": {"enum": list(PROPERTY_SIGNATURES)},
                    "object": {"type": "string", "pattern": r".*\S.*"},
                    "provenance": {
                        "type": "array",
                        "items": {"type": "string", "pattern": r".*\S.*"},
                        "uniqueItems": True,
                    },
                    "evidence": {"type": "string"},
                },
            },
        },
    },
}
STRICT_VALIDATOR = jsonschema.Draft202012Validator(STRICT_EXTRACTION_SCHEMA)


class OfflineAblationError(RuntimeError):
    """Raised when immutable evidence cannot support the requested run."""


class MappingReplayClient:
    """Return frozen responses by unique prompt; never access the network."""

    def __init__(self, responses: Mapping[str, str], *, fail: bool = False) -> None:
        self.responses = dict(responses)
        self.fail = fail
        self.call_count = 0

    async def generate_response(self, _system_prompt: str, prompt: str) -> str:
        self.call_count += 1
        if self.fail:
            raise OfflineAblationError("synthetic client failure fixture")
        if prompt not in self.responses:
            raise OfflineAblationError("unregistered offline replay prompt")
        return self.responses[prompt]


def extract_json_text(text: str) -> str:
    """Apply the same code-fence normalization as the production extractor."""

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return fenced.group(1) if fenced else text.strip()


def parse_only(text: str) -> tuple[bool, Any | None, list[str]]:
    """Run only JSON decoding; retain a stable diagnostic on failure."""

    try:
        return True, json.loads(extract_json_text(text)), []
    except (json.JSONDecodeError, TypeError) as exc:
        return False, None, [f"{type(exc).__name__}: {exc}"]


def strict_schema_validate(text: str) -> tuple[bool, Any | None, list[str]]:
    """Validate the decoded value using a strict Draft 2020-12 schema."""

    parsed, value, diagnostics = parse_only(text)
    if not parsed:
        return False, None, diagnostics
    errors = sorted(
        STRICT_VALIDATOR.iter_errors(value),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    rendered = [
        f"{'.'.join(map(str, error.absolute_path)) or '$'}: {error.message}"
        for error in errors
    ]
    return not rendered, value, rendered


def _payload(entities: list[dict[str, Any]], relations: list[dict[str, Any]]) -> str:
    return json.dumps({"entities": entities, "relations": relations}, sort_keys=True)


def build_structural_fixtures() -> list[dict[str, Any]]:
    """Return the pre-specified contract/failure ledger before any execution."""

    actor = {"id": "a", "type": "ThreatActor", "name": "Actor"}
    malware = {"id": "m", "type": "Malware", "name": "Malware"}
    vuln = {"id": "v", "type": "Vulnerability", "name": "CVE-TEST"}
    campaign = {"id": "c", "type": "Campaign", "name": "Campaign"}
    indicator = {"id": "i", "type": "Indicator", "name": "Indicator"}
    all_entities = [actor, malware, vuln, campaign, indicator]

    fixtures: list[dict[str, Any]] = []

    def add(
        case_id: str,
        family: str,
        responses: Sequence[str],
        *,
        known_violation: bool,
        expected_failure_stage: str | None,
        expected_derivations: int = 0,
        client_error: bool = False,
        note: str,
    ) -> None:
        fixtures.append(
            {
                "case_id": case_id,
                "source_kind": "synthetic_fixture",
                "family": family,
                "known_contract_violation": known_violation,
                "expected_failure_stage": expected_failure_stage,
                "expected_derivations": expected_derivations,
                "client_error": client_error,
                "responses": list(responses),
                "response_sha256": [
                    hashlib.sha256(item.encode("utf-8")).hexdigest()
                    for item in responses
                ],
                "note": note,
            }
        )

    add(
        "fixture-valid-isolated", "valid_control", [_payload([actor], [])],
        known_violation=False, expected_failure_stage=None,
        note="Smallest non-empty graph accepted by the production contract.",
    )
    add(
        "fixture-valid-swrl-chain", "valid_control",
        [_payload([actor, malware, vuln], [
            {"subject": "a", "predicate": "Uses", "object": "m"},
            {"subject": "m", "predicate": "Targets", "object": "v"},
        ])], known_violation=False, expected_failure_stage=None,
        expected_derivations=1,
        note="Licensed SWRL chain with one expected derived actor-target edge.",
    )
    add(
        "fixture-valid-type-aware-homonym", "identity_control",
        [_payload([
            {"id": "x1", "type": "ThreatActor", "name": "Shared Label"},
            {"id": "x2", "type": "Malware", "name": "Shared Label"},
        ], [])], known_violation=False, expected_failure_stage=None,
        note="Same normalized name in two types must remain two OWL individuals.",
    )
    add(
        "fixture-empty-graph", "empty", [_payload([], [])],
        known_violation=True, expected_failure_stage="nonempty_gate",
        note="Structurally valid but intentionally not a verified graph artifact.",
    )
    add(
        "fixture-malformed-json", "malformed", ['{"entities": ['],
        known_violation=True, expected_failure_stage="json_parse",
        note="Truncated JSON.",
    )
    add(
        "fixture-root-array", "root_shape", [json.dumps([])],
        known_violation=True, expected_failure_stage="json_schema",
        note="JSON root is not an extraction object.",
    )
    add(
        "fixture-missing-arrays", "root_shape", [json.dumps({"entities": []})],
        known_violation=True, expected_failure_stage="json_schema",
        note="Required relations array is missing.",
    )
    add(
        "fixture-unknown-root-key", "unknown_field",
        [json.dumps({"entities": [actor], "relations": [], "unexpected": True})],
        known_violation=True, expected_failure_stage="json_schema",
        note="Unknown root keys are rejected instead of being silently ignored.",
    )
    add(
        "fixture-unknown-entity-key", "unknown_field",
        [_payload([{**actor, "unexpected": True}], [])],
        known_violation=True, expected_failure_stage="json_schema",
        note="Unknown entity keys are rejected instead of being silently ignored.",
    )
    add(
        "fixture-unknown-relation-key", "unknown_field",
        [_payload([actor, malware], [{
            "subject": "a",
            "predicate": "Uses",
            "object": "m",
            "unexpected": True,
        }])],
        known_violation=True, expected_failure_stage="json_schema",
        note="Unknown relation keys are rejected instead of being silently ignored.",
    )
    add(
        "fixture-invalid-entity-type", "entity_schema",
        [_payload([{"id": "x", "type": "Tool", "name": "X"}], [])],
        known_violation=True, expected_failure_stage="json_schema",
        note="Entity type is outside the MALOnt extraction vocabulary.",
    )
    add(
        "fixture-blank-name", "entity_schema",
        [_payload([{"id": "x", "type": "Malware", "name": "  "}], [])],
        known_violation=True, expected_failure_stage="json_schema",
        note="Whitespace-only entity name.",
    )
    add(
        "fixture-invalid-aliases", "entity_schema",
        [_payload([{**malware, "aliases": [""]}], [])],
        known_violation=True, expected_failure_stage="json_schema",
        note="Alias array contains an empty string.",
    )
    add(
        "fixture-invalid-provenance", "entity_schema",
        [_payload([{**malware, "provenance": [7]}], [])],
        known_violation=True, expected_failure_stage="json_schema",
        note="Provenance is not an array of strings.",
    )
    add(
        "fixture-duplicate-entity-id", "referential_integrity",
        [_payload([malware, {"id": "m", "type": "Malware", "name": "Other"}], [])],
        known_violation=True, expected_failure_stage="production_contract",
        note="JSON Schema cannot express uniqueness of one object property across records.",
    )
    add(
        "fixture-invalid-predicate", "relation_schema",
        [_payload([actor, malware], [{"subject": "a", "predicate": "Runs", "object": "m"}])],
        known_violation=True, expected_failure_stage="json_schema",
        note="Predicate is outside the allowed vocabulary.",
    )
    add(
        "fixture-dangling-subject", "referential_integrity",
        [_payload([malware], [{"subject": "missing", "predicate": "Uses", "object": "m"}])],
        known_violation=True, expected_failure_stage="production_contract",
        note="Relation subject is not declared.",
    )
    add(
        "fixture-dangling-object", "referential_integrity",
        [_payload([actor], [{"subject": "a", "predicate": "Uses", "object": "missing"}])],
        known_violation=True, expected_failure_stage="production_contract",
        note="Relation object is not declared.",
    )

    wrong_signatures = [
        ("Uses", campaign, malware, "subject"),
        ("Uses", actor, campaign, "object"),
        ("Targets", actor, vuln, "subject"),
        ("Targets", malware, actor, "object"),
        ("AssociatedWith", actor, campaign, "subject"),
        ("AssociatedWith", malware, vuln, "object"),
        ("HasIndicator", actor, indicator, "subject"),
        ("HasIndicator", malware, campaign, "object"),
    ]
    for predicate, subject, obj, endpoint in wrong_signatures:
        add(
            f"fixture-{predicate.lower()}-wrong-{endpoint}",
            "type_signature",
            [_payload([subject, obj], [{
                "subject": subject["id"], "predicate": predicate, "object": obj["id"]
            }])],
            known_violation=True,
            expected_failure_stage="production_contract",
            note=f"{predicate} {endpoint} type violates the production signature.",
        )

    add(
        "fixture-duplicate-relation", "relation_uniqueness",
        [_payload([actor, malware], [
            {"subject": "a", "predicate": "Uses", "object": "m", "provenance": ["p1"]},
            {"subject": "a", "predicate": "Uses", "object": "m", "provenance": ["p2"]},
        ])], known_violation=True, expected_failure_stage="production_contract",
        note="Same semantic relation appears twice with different metadata.",
    )
    add(
        "fixture-partial-two-chunk", "aggregate_failure",
        [_payload([actor], []), '{"entities":'],
        known_violation=True, expected_failure_stage="aggregate_status",
        note="One successful and one malformed chunk must yield PARTIAL and block reasoning.",
    )
    add(
        "fixture-client-error", "client_failure", [],
        known_violation=True, expected_failure_stage="client_error", client_error=True,
        note="Synthetic local client exception; no provider is contacted.",
    )
    return fixtures


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise OfflineAblationError(f"{path}:{number} is not a JSON object")
        rows.append(value)
    return rows


def load_hosted_cases(source_run: Path) -> list[dict[str, Any]]:
    """Join frozen case metadata and raw response text without copying text."""

    cases_path = source_run / "cases.jsonl"
    raw_path = source_run / "raw_responses.jsonl"
    if not cases_path.is_file() or not raw_path.is_file():
        raise OfflineAblationError("hosted source cases/raw responses are missing")
    cases = {row["case_id"]: row for row in _read_jsonl(cases_path)}
    raw = {row["case_id"]: row for row in _read_jsonl(raw_path)}
    if set(cases) != set(raw):
        raise OfflineAblationError("hosted case and response IDs differ")
    output: list[dict[str, Any]] = []
    for case_id in sorted(cases):
        text = raw[case_id].get("text")
        if not isinstance(text, str):
            raise OfflineAblationError(f"hosted response {case_id} has no text")
        output.append(
            {
                "case_id": case_id,
                "source_kind": "frozen_hosted_response",
                "family": str(cases[case_id].get("condition", "unknown")),
                "known_contract_violation": None,
                "expected_failure_stage": None,
                "expected_derivations": 0,
                "client_error": False,
                "responses": [text],
                "response_sha256": [hashlib.sha256(text.encode("utf-8")).hexdigest()],
                "source_path": cases[case_id].get("source_path"),
                "source_row_index": cases[case_id].get("source_row_index"),
            }
        )
    return output


def _chunks_and_client(case: Mapping[str, Any]) -> tuple[list[StructuredChunk], MappingReplayClient]:
    responses = list(case.get("responses", ()))
    count = max(1, len(responses))
    prompts = [f"offline-observation:{case['case_id']}:{index}" for index in range(count)]
    response_map = {prompt: response for prompt, response in zip(prompts, responses)}
    chunks = [
        StructuredChunk.create(
            text=prompt,
            source_id=f"offline:{case['case_id']}",
            index=index,
            start_word=0,
            end_word=1,
        )
        for index, prompt in enumerate(prompts)
    ]
    return chunks, MappingReplayClient(response_map, fail=bool(case.get("client_error")))


def _raw_fact_counts(values: Sequence[Any]) -> tuple[int, int]:
    entities = 0
    relations = 0
    for value in values:
        if isinstance(value, Mapping):
            if isinstance(value.get("entities"), list):
                entities += len(value["entities"])
            if isinstance(value.get("relations"), list):
                relations += len(value["relations"])
    return entities, relations


def _production_provenance(extraction: Mapping[str, Any]) -> tuple[int, int]:
    registry = set(extraction.get("chunk_registry", {}))
    facts = list(extraction.get("entities", ())) + list(extraction.get("relations", ()))
    complete = sum(
        bool(fact.get("provenance"))
        and set(fact.get("provenance", ())).issubset(registry)
        for fact in facts
    )
    return complete, len(facts)


async def evaluate_observation(case: Mapping[str, Any], log_dir: Path) -> dict[str, Any]:
    """Evaluate one observation through all four cumulative capability stages."""

    started = time.perf_counter()
    responses = list(case.get("responses", ()))
    parse_results = [parse_only(text) for text in responses]
    schema_results = [strict_schema_validate(text) for text in responses]
    parse_accepted = bool(responses) and all(item[0] for item in parse_results)
    schema_accepted = bool(responses) and all(item[0] for item in schema_results)
    parsed_values = [item[1] for item in parse_results if item[0]]
    raw_entities, raw_relations = _raw_fact_counts(parsed_values)

    chunks, client = _chunks_and_client(case)
    extractor = MalOntExtractor(client)
    extraction = await extractor.process_document_chunks(chunks)
    canonical_accepted = extraction.status is ExtractionStatus.SUCCESS
    provenance_complete, provenance_total = _production_provenance(extraction)

    load_status = ReasoningStatus.NOT_RUN.value
    reasoning_status = ReasoningStatus.NOT_RUN.value
    inferred: list[dict[str, Any]] = []
    full_accepted = False
    reasoner_log = ""
    if canonical_accepted:
        reasoner = MalOntReasoner(
            ontology_iri=f"http://example.org/ablation/{case['case_id']}#"
        )
        load = reasoner.load_extracted_data(extraction)
        load_status = load.status.value
        if load.status is ReasoningStatus.VALIDATED:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                result = reasoner.verify_and_infer()
            reasoner_log = buffer.getvalue()
            reasoning_status = result.status.value
            inferred = list(result.get("inferred_relations", ()))
            full_accepted = result.status is ReasoningStatus.CONSISTENT
        reasoner.world.close()

    log_path = log_dir / f"{case['case_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(reasoner_log, encoding="utf-8")
    proof_complete = sum(
        bool(edge.get("proofs"))
        and all(proof.get("support_relation_ids") for proof in edge.get("proofs", ()))
        for edge in inferred
    )
    expected_derivations = int(case.get("expected_derivations", 0))
    failure_stage = None
    if not parse_accepted:
        failure_stage = "client_error" if case.get("client_error") else "json_parse"
    elif not schema_accepted:
        failure_stage = "json_schema"
    elif extraction.status is ExtractionStatus.PARTIAL:
        failure_stage = "aggregate_status"
    elif extraction.status is ExtractionStatus.EMPTY:
        failure_stage = "nonempty_gate"
    elif extraction.status is not ExtractionStatus.SUCCESS:
        failure_stage = "production_contract"
    elif load_status != ReasoningStatus.VALIDATED.value:
        failure_stage = "ontology_load"
    elif reasoning_status != ReasoningStatus.CONSISTENT.value:
        failure_stage = "pellet"

    methods = {
        "json_parse_only": {
            "accepted": parse_accepted,
            "diagnostics": [item for result in parse_results for item in result[2]],
            "fact_count": raw_entities + raw_relations,
            "provenance_complete": 0,
            "provenance_total": raw_entities + raw_relations,
            "derivations_with_complete_proof": 0,
        },
        "strict_json_schema": {
            "accepted": schema_accepted,
            "diagnostics": [item for result in schema_results for item in result[2]],
            "fact_count": raw_entities + raw_relations,
            "provenance_complete": 0,
            "provenance_total": raw_entities + raw_relations,
            "derivations_with_complete_proof": 0,
        },
        "production_canonicalization": {
            "accepted": canonical_accepted,
            "status": extraction.status.value,
            "diagnostics": list(extraction.get("diagnostics", ())),
            "fact_count": provenance_total,
            "provenance_complete": provenance_complete,
            "provenance_total": provenance_total,
            "derivations_with_complete_proof": 0,
        },
        "full_automalkg": {
            "accepted": full_accepted,
            "load_status": load_status,
            "reasoning_status": reasoning_status,
            "inferred_relation_count": len(inferred),
            "inferred_relations_with_complete_proof": proof_complete,
            "fact_count": provenance_total,
            "provenance_complete": provenance_complete if full_accepted else 0,
            "provenance_total": provenance_total if full_accepted else 0,
            "derivations_with_complete_proof": min(proof_complete, expected_derivations),
        },
    }
    return {
        "case_id": case["case_id"],
        "source_kind": case["source_kind"],
        "family": case["family"],
        "known_contract_violation": case.get("known_contract_violation"),
        "expected_failure_stage": case.get("expected_failure_stage"),
        "expected_derivations": expected_derivations,
        "response_sha256": case.get("response_sha256"),
        "hosted_call_attempts": 0,
        "offline_client_calls": client.call_count,
        "observed_failure_stage": failure_stage,
        "methods": methods,
        "runtime_seconds": time.perf_counter() - started,
        "reasoner_log": {
            "path": log_path.relative_to(PAPER_ROOT).as_posix(),
            "sha256": sha256_file(log_path),
        },
    }


def _valid_load(reasoner: MalOntReasoner, entities: list[dict[str, str]], relations: list[dict[str, str]] | None = None) -> None:
    result = reasoner.load_extracted_data(
        {"status": "success", "entities": entities, "relations": relations or []}
    )
    if result.status is not ReasoningStatus.VALIDATED:
        raise OfflineAblationError(f"integration fixture load failed: {result}")


def _run_reasoner_logged(reasoner: MalOntReasoner, log_path: Path) -> dict[str, Any]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        result = reasoner.verify_and_infer()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(buffer.getvalue(), encoding="utf-8")
    output = dict(result)
    output["log"] = {
        "path": log_path.relative_to(PAPER_ROOT).as_posix(),
        "sha256": sha256_file(log_path),
    }
    reasoner.world.close()
    return output


def run_reachability_fixtures(log_dir: Path) -> list[dict[str, Any]]:
    """Exercise OWL axioms and execution failures at their actual reachable layer."""

    rows: list[dict[str, Any]] = []
    for first, second in itertools.combinations(ENTITY_TYPES, 2):
        fixture_id = f"owl-disjoint-{first.lower()}-{second.lower()}"
        reasoner = MalOntReasoner(f"http://example.org/reach/{fixture_id}#")
        _valid_load(reasoner, [{"id": "x", "type": first, "name": "X"}])
        reasoner.entity_map["x"].is_a.append(getattr(reasoner.onto, second))
        observed = _run_reasoner_logged(reasoner, log_dir / f"{fixture_id}.log")
        rows.append({
            "fixture_id": fixture_id,
            "axiom_or_failure_family": "AllDisjoint",
            "member": f"{first} vs {second}",
            "production_input_reachable": False,
            "reasoner_layer_injection": True,
            "expected_status": ReasoningStatus.INCONSISTENT.value,
            "observed_status": observed["status"],
            "passed": observed["status"] == ReasoningStatus.INCONSISTENT.value,
            "note": "The one-type-per-ID contract and type-aware identity prevent this dual typing end to end.",
            "log": observed["log"],
        })

    incompatible = {
        ("Uses", "domain"): ("Malware", "Malware"),
        ("Uses", "range"): ("ThreatActor", "ThreatActor"),
        ("Targets", "domain"): ("ThreatActor", "Vulnerability"),
        ("Targets", "range"): ("Malware", "Malware"),
        ("AssociatedWith", "domain"): ("ThreatActor", "Campaign"),
        ("AssociatedWith", "range"): ("Malware", "Vulnerability"),
        ("HasIndicator", "domain"): ("ThreatActor", "Indicator"),
        ("HasIndicator", "range"): ("Malware", "Campaign"),
    }
    for (predicate, endpoint), (subject_type, object_type) in incompatible.items():
        fixture_id = f"owl-{predicate.lower()}-{endpoint}"
        reasoner = MalOntReasoner(f"http://example.org/reach/{fixture_id}#")
        _valid_load(reasoner, [
            {"id": "s", "type": subject_type, "name": "Subject"},
            {"id": "o", "type": object_type, "name": "Object"},
        ])
        getattr(reasoner.entity_map["s"], predicate).append(reasoner.entity_map["o"])
        observed = _run_reasoner_logged(reasoner, log_dir / f"{fixture_id}.log")
        expected = (
            ReasoningStatus.CONSISTENT.value
            if predicate == "Targets" and endpoint == "domain"
            else ReasoningStatus.INCONSISTENT.value
        )
        rows.append({
            "fixture_id": fixture_id,
            "axiom_or_failure_family": "ObjectPropertyDomainRange",
            "member": f"{predicate}.{endpoint}",
            "production_input_reachable": False,
            "reasoner_layer_injection": True,
            "expected_status": expected,
            "observed_status": observed["status"],
            "passed": observed["status"] == expected,
            "note": (
                "Targets has OWL domain Entity, so this is non-discriminating; the extraction signature is intentionally stricter."
                if predicate == "Targets" and endpoint == "domain"
                else "Production relation signatures reject this endpoint typing before ontology mutation."
            ),
            "log": observed["log"],
        })

    swrl_cases = {
        "swrl-positive": [
            {"subject": "a", "predicate": "Uses", "object": "m", "provenance": ["fixture:uses"]},
            {"subject": "m", "predicate": "Targets", "object": "v", "provenance": ["fixture:targets"]},
        ],
        "swrl-missing-uses": [
            {"subject": "m", "predicate": "Targets", "object": "v", "provenance": ["fixture:targets"]},
        ],
        "swrl-missing-targets": [
            {"subject": "a", "predicate": "Uses", "object": "m", "provenance": ["fixture:uses"]},
        ],
        "swrl-disconnected": [
            {"subject": "a", "predicate": "Uses", "object": "m", "provenance": ["fixture:uses"]},
            {"subject": "d", "predicate": "Targets", "object": "v", "provenance": ["fixture:targets"]},
        ],
    }
    for fixture_id, relations in swrl_cases.items():
        entities = [
            {"id": "a", "type": "ThreatActor", "name": "Actor"},
            {"id": "m", "type": "Malware", "name": "Malware"},
            {"id": "d", "type": "Malware", "name": "Decoy"},
            {"id": "v", "type": "Vulnerability", "name": "Vulnerability"},
        ]
        reasoner = MalOntReasoner(f"http://example.org/reach/{fixture_id}#")
        _valid_load(reasoner, entities, relations)
        observed = _run_reasoner_logged(reasoner, log_dir / f"{fixture_id}.log")
        inferred = observed.get("inferred_relations", [])
        expected_count = 1 if fixture_id == "swrl-positive" else 0
        proof_count = sum(bool(edge.get("proofs")) for edge in inferred)
        rows.append({
            "fixture_id": fixture_id,
            "axiom_or_failure_family": "SWRL",
            "member": fixture_id.removeprefix("swrl-"),
            "production_input_reachable": True,
            "reasoner_layer_injection": False,
            "expected_status": ReasoningStatus.CONSISTENT.value,
            "observed_status": observed["status"],
            "expected_inferred_relations": expected_count,
            "observed_inferred_relations": len(inferred),
            "proof_complete_relations": proof_count,
            "passed": (
                observed["status"] == ReasoningStatus.CONSISTENT.value
                and len(inferred) == expected_count
                and proof_count == expected_count
            ),
            "note": "Positive and missing-body controls are valid end-to-end inputs.",
            "log": observed["log"],
        })

    execution_fixtures = [
        ("reasoning-not-run", ReasoningStatus.NOT_RUN, None),
        ("reasoning-validation-error", ReasoningStatus.VALIDATION_ERROR, None),
        ("reasoner-unavailable", ReasoningStatus.REASONER_UNAVAILABLE, FileNotFoundError("synthetic missing Java")),
        ("reasoner-error", ReasoningStatus.REASONER_ERROR, RuntimeError("synthetic execution fault")),
    ]
    for fixture_id, expected_status, injected in execution_fixtures:
        reasoner = MalOntReasoner(f"http://example.org/reach/{fixture_id}#")
        if expected_status is ReasoningStatus.NOT_RUN:
            result = reasoner.verify_and_infer()
        elif expected_status is ReasoningStatus.VALIDATION_ERROR:
            result = reasoner.load_extracted_data({"entities": [], "relations": []})
        else:
            _valid_load(reasoner, [{"id": "m", "type": "Malware", "name": "M"}])
            original = reasoning_module.sync_reasoner_pellet
            def _raise(*_args: Any, _error: Exception = injected, **_kwargs: Any) -> None:
                raise _error
            reasoning_module.sync_reasoner_pellet = _raise
            try:
                result = reasoner.verify_and_infer()
            finally:
                reasoning_module.sync_reasoner_pellet = original
        reasoner.world.close()
        rows.append({
            "fixture_id": fixture_id,
            "axiom_or_failure_family": "ReasoningStatus",
            "member": expected_status.value,
            "production_input_reachable": expected_status in {ReasoningStatus.REASONER_UNAVAILABLE, ReasoningStatus.REASONER_ERROR},
            "reasoner_layer_injection": injected is not None,
            "expected_status": expected_status.value,
            "observed_status": result.status.value,
            "passed": result.status is expected_status,
            "note": "Execution failures use explicit local fault injection; no hosted or external failure is claimed.",
            "log": None,
        })
    return rows


def summarize_observations(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate acceptance and evidence metrics from observation-level rows."""

    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        by_source[str(row["source_kind"])].append(row)

    summaries: dict[str, Any] = {}
    for source_kind, rows in sorted(by_source.items()):
        method_metrics: dict[str, Any] = {}
        known = [row for row in rows if row.get("known_contract_violation") is True]
        expected_derivations = sum(int(row.get("expected_derivations", 0)) for row in rows)
        for method in METHODS:
            accepted = sum(bool(row["methods"][method]["accepted"]) for row in rows)
            silent = sum(bool(row["methods"][method]["accepted"]) for row in known)
            provenance_complete = sum(int(row["methods"][method].get("provenance_complete", 0)) for row in rows)
            provenance_total = sum(int(row["methods"][method].get("provenance_total", 0)) for row in rows)
            proof_complete = sum(int(row["methods"][method].get("derivations_with_complete_proof", 0)) for row in rows)
            method_metrics[method] = {
                "accepted": accepted,
                "rejected": len(rows) - accepted,
                "acceptance_rate": accepted / len(rows) if rows else None,
                "known_violation_cases": len(known),
                "silent_acceptance_of_known_violations": silent,
                "silent_acceptance_rate": silent / len(known) if known else None,
                "resolvable_provenance_facts": provenance_complete,
                "emitted_facts_for_provenance": provenance_total,
                "provenance_completeness": provenance_complete / provenance_total if provenance_total else None,
                "expected_derivations": expected_derivations,
                "derivations_with_complete_proof": proof_complete,
                "proof_ledger_capability_coverage": proof_complete / expected_derivations if expected_derivations else None,
            }
        summaries[source_kind] = {
            "case_count": len(rows),
            "methods": method_metrics,
            "observed_failure_stage_distribution": dict(sorted(Counter(str(row.get("observed_failure_stage")) for row in rows).items())),
        }
    return summaries


def summarize_reachability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    inconsistency_rows = [
        row for row in rows
        if row["expected_status"] == ReasoningStatus.INCONSISTENT.value
    ]
    disjoint = [row for row in inconsistency_rows if row["axiom_or_failure_family"] == "AllDisjoint"]
    restrictions = [row for row in inconsistency_rows if row["axiom_or_failure_family"] == "ObjectPropertyDomainRange"]
    return {
        "row_count": len(rows),
        "passed": sum(bool(row["passed"]) for row in rows),
        "failed": sum(not bool(row["passed"]) for row in rows),
        "owl_inconsistency_classes": len(inconsistency_rows),
        "owl_inconsistency_classes_reachable_from_production_input": sum(bool(row["production_input_reachable"]) for row in inconsistency_rows),
        "owl_inconsistency_classes_reachable_by_reasoner_integration_injection": sum(bool(row["reasoner_layer_injection"]) for row in inconsistency_rows),
        "disjoint_class_pairs_tested": len(disjoint),
        "property_domain_range_inconsistencies_tested": len(restrictions),
        "non_discriminating_axioms": [
            row["member"] for row in rows
            if row["axiom_or_failure_family"] == "ObjectPropertyDomainRange"
            and row["expected_status"] == ReasoningStatus.CONSISTENT.value
        ],
    }


def build_gates(
    observations: Sequence[Mapping[str, Any]],
    reachability: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    *,
    fixtures_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixtures = [row for row in observations if row["source_kind"] == "synthetic_fixture"]
    known = [row for row in fixtures if row["known_contract_violation"] is True]
    full_silent = sum(row["methods"]["full_automalkg"]["accepted"] for row in known)
    reach_failures = sum(not row["passed"] for row in reachability)
    g4_failed: list[str] = []
    if full_silent:
        g4_failed.append("full_pipeline_silently_accepts_known_violation")
    if reach_failures:
        g4_failed.append("reachability_fixture_mismatch")
    if metrics["hosted_call_attempts"] != 0:
        g4_failed.append("hosted_call_attempted")
    g4 = {
        "gate_id": "G4_offline_stage_ablation_validity",
        "stage": "ablation",
        "verdict": "PASS" if not g4_failed else "FAIL_REPAIR",
        "hypothesis_or_mechanism": "Cumulative local stages isolate validation, provenance, reasoning, and proof-ledger capabilities on identical inputs.",
        "input_artifacts": ["fixtures.jsonl", "observations.jsonl", "stage_reachability_matrix.jsonl"],
        "command_or_check": "deterministic ledger reconciliation in build_gates",
        "pass_criteria": {
            "full_pipeline_silent_acceptance": 0,
            "reachability_fixture_failures": 0,
            "hosted_call_attempts": 0,
        },
        "metrics": {
            "known_violation_fixtures": len(known),
            "full_pipeline_silent_acceptance": full_silent,
            "reachability_fixture_failures": reach_failures,
            "hosted_call_attempts": metrics["hosted_call_attempts"],
        },
        "failed_criteria": g4_failed,
        "fail_interpretation": "The staged comparison does not isolate the intended contract or a semantic fixture disagrees with its pre-specified expectation.",
        "next_allowed_step": "bounded offline ablation reporting",
    }
    case_ids = [row["case_id"] for row in observations]
    hosted_count = sum(
        row["source_kind"] == "frozen_hosted_response" for row in observations
    )
    expected_hosted_count = 0 if fixtures_only else EXPECTED_HOSTED_CASES
    g6_failed: list[str] = []
    if len(case_ids) != len(set(case_ids)):
        g6_failed.append("duplicate_observation_id")
    if hosted_count != expected_hosted_count:
        g6_failed.append(
            f"hosted_case_count_not_{expected_hosted_count}_for_execution_scope"
        )
    if any(row["hosted_call_attempts"] != 0 for row in observations):
        g6_failed.append("nonzero_hosted_calls")
    if any(not row.get("response_sha256") and row["source_kind"] != "synthetic_fixture" for row in observations):
        g6_failed.append("missing_source_hash")
    g6 = {
        "gate_id": "G6_offline_stage_ablation_result_integrity",
        "stage": "results",
        "verdict": "PASS" if not g6_failed else "FAIL_REPAIR",
        "hypothesis_or_mechanism": "Every reported count is reconstructable from unique observation rows tied to frozen source hashes.",
        "input_artifacts": ["observations.jsonl", "metrics.json"],
        "command_or_check": "deterministic identity/count/hash reconciliation in build_gates",
        "pass_criteria": {
            "unique_case_ids": True,
            "frozen_hosted_cases": expected_hosted_count,
            "hosted_call_attempts": 0,
            "source_hash_per_included_response": True,
        },
        "metrics": {
            "observation_count": len(observations),
            "unique_case_ids": len(set(case_ids)),
            "execution_scope": "fixtures_only" if fixtures_only else "hosted_and_fixtures",
            "expected_hosted_case_count": expected_hosted_count,
            "hosted_case_count": hosted_count,
            "hosted_call_attempts": metrics["hosted_call_attempts"],
        },
        "failed_criteria": g6_failed,
        "fail_interpretation": "The ablation table is incomplete, duplicated, untraceable, or not demonstrably offline.",
        "next_allowed_step": "bounded manuscript integration",
    }
    return g4, g6


def build_baseline_prerequisite_gates(
    observations: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Certify input parity/baseline sanity and bound fairness claims."""

    method_set_ok = all(set(row.get("methods", {})) == set(METHODS) for row in observations)
    source_hashes_present = all(
        bool(row.get("response_sha256"))
        or (
            row.get("source_kind") == "synthetic_fixture"
            and row.get("expected_failure_stage") == "client_error"
        )
        for row in observations
    )
    g1_failed = []
    if not method_set_ok:
        g1_failed.append("methods_not_evaluated_on_identical_observation_rows")
    if not source_hashes_present:
        g1_failed.append("missing_frozen_response_hash")
    g1 = {
        "gate_id": "G1_offline_ablation_input_parity",
        "stage": "preprocessing_to_baseline",
        "verdict": "PASS" if not g1_failed else "FAIL_REPAIR",
        "input_artifacts": ["fixtures.jsonl", "observations.jsonl"],
        "hypothesis_or_mechanism": "All four methods receive the same response bytes and case identities; only the enabled validation/reasoning layer changes.",
        "command_or_check": "observation-level method-set and source-hash reconciliation",
        "pass_criteria": {"method_set_per_observation": list(METHODS), "source_hashes_present": True},
        "metrics": {"observation_count": len(observations), "method_set_ok": method_set_ok, "source_hashes_present": source_hashes_present},
        "failed_criteria": g1_failed,
        "fail_interpretation": "The staged comparison changed its inputs or omitted a comparator for some cases.",
        "next_allowed_step": "baseline sanity evaluation",
    }

    fixture_methods = metrics["summaries"]["synthetic_fixture"]["methods"]
    g2_failed = []
    for method in METHODS:
        accepted = int(fixture_methods[method]["accepted"])
        rejected = int(fixture_methods[method]["rejected"])
        if accepted == 0 or rejected == 0:
            g2_failed.append(f"degenerate_acceptance_distribution:{method}")
    if fixture_methods["full_automalkg"]["silent_acceptance_of_known_violations"] != 0:
        g2_failed.append("full_pipeline_accepts_known_violation")
    g2 = {
        "gate_id": "G2_offline_baseline_sanity",
        "stage": "baseline",
        "verdict": "PASS" if not g2_failed else "FAIL_REPAIR",
        "input_artifacts": ["observations.jsonl", "metrics.json"],
        "hypothesis_or_mechanism": "Each cumulative comparator accepts valid controls and rejects at least one failure fixture; the full pipeline rejects every pre-specified violation.",
        "command_or_check": "deterministic fixture acceptance-distribution check",
        "pass_criteria": {"accepted_per_method": ">0", "rejected_per_method": ">0", "full_pipeline_silent_acceptance": 0},
        "metrics": {
            method: {"accepted": fixture_methods[method]["accepted"], "rejected": fixture_methods[method]["rejected"]}
            for method in METHODS
        },
        "failed_criteria": g2_failed,
        "fail_interpretation": "A comparator is degenerate on the failure/control ledger or the full contract silently admits a known violation.",
        "next_allowed_step": "proposed-method and ablation reporting",
    }
    hosted_count = sum(
        row.get("source_kind") == "frozen_hosted_response" for row in observations
    )
    threat_model = (
        "PASS: identical pre-specified contract violations and frozen hosted responses"
        if hosted_count
        else "PASS: identical pre-specified synthetic contract and mechanism fixtures; no hosted inputs included"
    )
    fairness = {
        "audit": "baseline_fairness",
        "verdict": "PASS_WITH_WARNINGS" if not (g1_failed or g2_failed) else "BLOCKED",
        "comparison_type": "cumulative offline capability ablation; not a performance leaderboard",
        "dimensions": {
            "same_data": "PASS: identical observation rows and response hashes",
            "same_preprocessing": "PASS: parse-only and JSON Schema use production-equivalent code-fence normalization",
            "same_metrics": "PASS: acceptance, silent violation, provenance, and proof definitions are shared",
            "tuning_budget": "NOT_APPLICABLE: deterministic validators have no tuned hyperparameters",
            "compute_resources": "NOT_COMPARABLE: later stages invoke Pellet; runtime superiority claims are prohibited",
            "threat_model": threat_model,
            "implementation_status": "WARNING: JSON Schema is a local nearest alternative, not provider-native constrained decoding",
        },
        "scientific_gates": {
            "G1_input_parity": g1["verdict"],
            "G2_baseline_sanity": g2["verdict"],
            "G4_ablation_validity": "see G4_ablation_validity.json",
            "G6_result_integrity": "see G6_result_integrity.json",
        },
        "allowed_claim_boundary": "component capability, failure localization, provenance completeness, proof-ledger coverage, and production reachability",
        "blocked_claims": [
            "provider-native structured-output performance",
            "factual extraction accuracy",
            "statistical or latency superiority",
        ],
    }
    return g1, g2, fairness


async def execute(
    source_run: Path,
    output_dir: Path,
    java: str | Path | None,
    *,
    fixtures_only: bool = False,
) -> dict[str, Any]:
    source_run = source_run.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise OfflineAblationError(f"refusing to overwrite run directory: {output_dir}")
    java_executable, java_audit = resolve_java(java)
    if java_executable is None:
        raise OfflineAblationError("Pellet requires a functioning Java executable")
    owl.JAVA_EXE = str(java_executable)
    output_dir.mkdir(parents=True)
    started_at = utc_now()

    fixtures = build_structural_fixtures()
    if len(fixtures) != EXPECTED_SYNTHETIC_FIXTURES:
        raise OfflineAblationError(
            f"fixture ledger drift: expected {EXPECTED_SYNTHETIC_FIXTURES}, found {len(fixtures)}"
        )
    known_violation_count = sum(
        row["known_contract_violation"] is True for row in fixtures
    )
    if known_violation_count != EXPECTED_KNOWN_VIOLATIONS:
        raise OfflineAblationError(
            "fixture violation ledger drift: "
            f"expected {EXPECTED_KNOWN_VIOLATIONS}, found {known_violation_count}"
        )
    fixture_public = [
        {key: value for key, value in row.items() if key != "responses"}
        for row in fixtures
    ]
    fixtures_path = output_dir / "fixtures.jsonl"
    write_jsonl(fixtures_path, fixture_public)

    hosted = [] if fixtures_only else load_hosted_cases(source_run)
    observations: list[dict[str, Any]] = []
    log_dir = output_dir / "pellet_logs"
    for case in [*hosted, *fixtures]:
        observations.append(await evaluate_observation(case, log_dir))
    reachability = run_reachability_fixtures(log_dir)

    observations_path = output_dir / "observations.jsonl"
    reachability_path = output_dir / "stage_reachability_matrix.jsonl"
    write_jsonl(observations_path, observations)
    write_jsonl(reachability_path, reachability)
    metrics = {
        "run_id": FIXTURE_ONLY_RUN_ID if fixtures_only else RUN_ID,
        "study_type": (
            "offline_fixture_only_stage_reachability"
            if fixtures_only
            else "offline_nearest_alternative_ablation"
        ),
        "execution_scope": "fixtures_only" if fixtures_only else "hosted_and_fixtures",
        "included_source_kinds": sorted({row["source_kind"] for row in observations}),
        "hosted_input_files_read": not fixtures_only,
        "provider_native_structured_output_used": False,
        "hosted_call_attempts": 0,
        "observation_count": len(observations),
        "summaries": summarize_observations(observations),
        "reachability": summarize_reachability(reachability),
        "limitations": [
            "The JSON Schema comparator is an offline nearest alternative, not provider-native constrained decoding.",
            (
                "No hosted responses are included; fixture-only outcomes do not estimate factual extraction accuracy."
                if fixtures_only
                else "Hosted observations have no factual entity/relation gold labels, so acceptance is not extraction accuracy."
            ),
            "Direct OWL contradiction fixtures test the reasoner layer after deliberate bypass of the production extraction contract.",
            "The strict schema rejects unknown fields and checks structural shape, but JSON Schema does not enforce referential integrity or predicate type signatures.",
            "No latency superiority inference is licensed by this sequential local run.",
        ],
    }
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)
    g4, g6 = build_gates(
        observations, reachability, metrics, fixtures_only=fixtures_only
    )
    g1, g2, fairness = build_baseline_prerequisite_gates(observations, metrics)
    g1_path = output_dir / "G1_input_parity.json"
    g2_path = output_dir / "G2_baseline_sanity.json"
    g4_path = output_dir / "G4_ablation_validity.json"
    g6_path = output_dir / "G6_result_integrity.json"
    fairness_path = output_dir / "baseline_fairness_audit.json"
    write_json(g1_path, g1)
    write_json(g2_path, g2)
    write_json(g4_path, g4)
    write_json(g6_path, g6)
    write_json(fairness_path, fairness)

    freeze_path = output_dir / "pip_freeze.txt"
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=False, capture_output=True, text=True
    )
    freeze_path.write_text(completed.stdout, encoding="utf-8")
    source_snapshot = output_dir / "executed_run_stage_reachability_ablation.py"
    shutil.copy2(Path(__file__), source_snapshot)
    readme_path = output_dir / "README.md"
    fixture_summary = metrics["summaries"]["synthetic_fixture"]["methods"]
    reach = metrics["reachability"]
    if fixtures_only:
        scope_intro = (
            "This is a **fixture-only offline capability ablation**. It reads no "
            "hosted cases or raw responses and makes zero hosted calls."
        )
        replay_command = (
            "python3 src/run_stage_reachability_ablation.py --fixtures-only "
            "--output-dir reproduced_runs/reachability"
        )
        hosted_section = [
            "## Hosted-input scope",
            "",
            "No hosted inputs were read or evaluated in this execution scope.",
            "",
        ]
    else:
        hosted_summary = metrics["summaries"]["frozen_hosted_response"]["methods"]
        scope_intro = (
            "This is an **offline nearest-alternative ablation** over frozen hosted "
            "outputs. It did not use provider-native structured output and made zero hosted calls."
        )
        replay_command = (
            "python3 src/run_stage_reachability_ablation.py --output-dir "
            "reproduced_runs/<new-run-id>"
        )
        hosted_section = [
            "## Frozen hosted-response acceptance",
            "",
            "| Method | Accepted | Rejected |",
            "|---|---:|---:|",
            *[
                f"| {method} | {hosted_summary[method]['accepted']} | {hosted_summary[method]['rejected']} |"
                for method in METHODS
            ],
            "",
            "These are contract/stage outcomes, not factual extraction scores.",
            "",
        ]
    readme_path.write_text(
        "\n".join([
            "# Offline stage ablation and OWL reachability",
            "",
            scope_intro,
            "",
            "## Exact replay",
            "",
            "```bash",
            replay_command,
            "```",
            "",
            "The runner refuses to overwrite an existing directory. `manifest.json` hashes the source ledgers, observations, gates, logs, environment freeze, and executed source snapshot.",
            "",
            "## Fixture comparison",
            "",
            "| Method | Accepted | Rejected | Silent known violations | Provenance completeness | Proof capability |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {method} | {fixture_summary[method]['accepted']} | {fixture_summary[method]['rejected']} | {fixture_summary[method]['silent_acceptance_of_known_violations']} | {fixture_summary[method]['provenance_completeness']} | {fixture_summary[method]['proof_ledger_capability_coverage']} |"
                for method in METHODS
            ],
            "",
            *hosted_section,
            "## Reachability result",
            "",
            f"The current OWL vocabulary has {reach['owl_inconsistency_classes']} tested inconsistency classes: {reach['owl_inconsistency_classes_reachable_from_production_input']} are reachable from schema-valid/type-aware production input, while {reach['owl_inconsistency_classes_reachable_by_reasoner_integration_injection']} are reachable under explicit reasoner-layer injection. The broad `Targets.domain = Entity` axiom is non-discriminating and remains consistent for a ThreatActor subject; the production extraction signature is stricter.",
            "",
            "## Claim boundary",
            "",
            "The evidence supports failure-localization and mechanism-coverage statements only. It does not establish provider-native constrained-output performance, factual CTI accuracy, statistical superiority, or production latency.",
            "",
            f"Scientific gates: G1={g1['verdict']}, G2={g2['verdict']}, G4={g4['verdict']}, G6={g6['verdict']}. Baseline-fairness verdict: {fairness['verdict']} (capability claims only).",
            "",
        ]),
        encoding="utf-8",
    )

    finished_at = utc_now()
    output_files = [path for path in output_dir.rglob("*") if path.is_file() and path.name != "manifest.json"]
    inputs = [
        Path(__file__),
        PAPER_ROOT / "src" / "extraction.py",
        PAPER_ROOT / "src" / "reasoning.py",
        PAPER_ROOT / "src" / "contracts.py",
    ]
    if not fixtures_only:
        inputs[:0] = [
            source_run / "cases.jsonl",
            source_run / "raw_responses.jsonl",
        ]
    manifest = build_manifest(
        run_id=FIXTURE_ONLY_RUN_ID if fixtures_only else RUN_ID,
        command=sys.argv,
        seed=SEED,
        started_at=started_at,
        finished_at=finished_at,
        inputs=inputs,
        outputs=output_files,
        base=PAPER_ROOT,
        java=java_audit,
        dependencies=("owlready2", "jsonschema"),
        extra={
            "hosted_call_attempts": 0,
            "execution_scope": "fixtures_only" if fixtures_only else "hosted_and_fixtures",
            "hosted_input_files_read": not fixtures_only,
            "provider_native_structured_output_used": False,
            "immutability": "runner refuses overwrite; SHA-256 covers every material input/output except manifest self-hash",
        },
    )
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return {
        "output_dir": str(output_dir),
        "metrics": metrics,
        "g4": g4["verdict"],
        "g6": g6["verdict"],
        "g1": g1["verdict"],
        "g2": g2["verdict"],
        "baseline_fairness": fairness["verdict"],
        "manifest_sha256": sha256_file(manifest_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--java", type=Path)
    parser.add_argument(
        "--fixtures-only",
        action="store_true",
        help="run only synthetic contract, reachability, SWRL, and local fault fixtures; do not read frozen hosted inputs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = asyncio.run(
            execute(
                args.source_run,
                args.output_dir,
                args.java,
                fixtures_only=args.fixtures_only,
            )
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

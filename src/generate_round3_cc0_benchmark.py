#!/usr/bin/env python3
"""Validate and freeze the author-created Round-3 CC0 benchmark."""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiment_support import PAPER_ROOT, sha256_file, utc_now, write_json, write_jsonl
from run_hosted_stress import evaluate_response, stable_hash


EXPECTED_CONTEXT_COUNT = 16
EXPECTED_NEGATIVE_CONTROLS = 2
DEFAULT_SOURCE = (
    PAPER_ROOT
    / "benchmark" / "round3_cc0"
    / "author_source.json"
)
DEFAULT_OUTPUT = DEFAULT_SOURCE.parent
ENTITY_TYPES = {"ThreatActor", "Malware", "Vulnerability", "Campaign", "Indicator"}
RELATION_SIGNATURES = {
    ("ThreatActor", "Uses", "Malware"),
    ("Malware", "Targets", "Vulnerability"),
    ("Malware", "AssociatedWith", "Campaign"),
    ("Malware", "HasIndicator", "Indicator"),
}
DOCUMENTATION_NETWORKS = tuple(
    ipaddress.ip_network(value) for value in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")
)
IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


class BenchmarkSourceError(RuntimeError):
    """Raised when authored benchmark source violates the frozen design."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _relation_signatures(gold: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    id_to_type = {str(row["id"]): str(row["type"]) for row in gold["entities"]}
    return {
        (id_to_type[str(row["subject"])], str(row["predicate"]), id_to_type[str(row["object"])])
        for row in gold["relations"]
    }


def validate_source(source: Mapping[str, Any]) -> dict[str, Any]:
    """Validate authorship, licensing, privacy bounds, schema, and coverage."""

    failures: list[str] = []
    if source.get("license") != "CC0-1.0":
        failures.append("license must be CC0-1.0")
    for key in ("authorship_statement", "privacy_statement", "injection_note", "attack_canary"):
        if not isinstance(source.get(key), str) or not str(source[key]).strip():
            failures.append(f"missing non-empty {key}")
    contexts = source.get("contexts")
    if not isinstance(contexts, list) or len(contexts) != EXPECTED_CONTEXT_COUNT:
        failures.append(f"exactly {EXPECTED_CONTEXT_COUNT} contexts are required")
        contexts = contexts if isinstance(contexts, list) else []

    ids: list[str] = []
    texts: list[str] = []
    context_hashes: list[str] = []
    covered_types: set[str] = set()
    covered_signatures: set[tuple[str, str, str]] = set()
    gold_entity_count = 0
    gold_relation_count = 0
    negative_control_count = 0
    canary = str(source.get("attack_canary", ""))
    injection_note = str(source.get("injection_note", ""))
    if canary and canary not in injection_note:
        failures.append("injection note must contain the exact attack canary")

    for index, row in enumerate(contexts):
        label = f"contexts[{index}]"
        if not isinstance(row, Mapping):
            failures.append(f"{label} must be an object")
            continue
        context_id = str(row.get("context_id", ""))
        text = str(row.get("text", ""))
        if not re.fullmatch(r"r3c\d{2}", context_id):
            failures.append(f"{label} has invalid context_id")
        if not text or text != " ".join(text.split()):
            failures.append(f"{label} text must be non-empty normalized whitespace")
        if canary and canary in text:
            failures.append(f"{label} clean text contains the injection canary")
        ids.append(context_id)
        texts.append(text)
        context_hashes.append(stable_hash(text))

        for match in IPV4_PATTERN.findall(text):
            address = ipaddress.ip_address(match)
            if not any(address in network for network in DOCUMENTATION_NETWORKS):
                failures.append(f"{label} uses non-documentation IPv4 address {match}")
        domains = re.findall(r"\b(?:[A-Za-z0-9-]+\.)+(?:example|invalid)\b", text)
        if "@" in text:
            failures.append(f"{label} contains an email-like token")
        if re.search(r"\bCVE-(?!2099-)\d{4}-\d+\b", text):
            failures.append(f"{label} contains a non-fictitious-year CVE-like identifier")

        gold = row.get("gold")
        if not isinstance(gold, Mapping):
            failures.append(f"{label} lacks a gold object")
            continue
        negative_control = bool(row.get("negative_control", False))
        gold_empty = bool(row.get("gold_empty", False))
        if negative_control:
            negative_control_count += 1
            if not gold_empty:
                failures.append(f"{label} negative_control must set gold_empty=true")
            if gold.get("entities") != [] or gold.get("relations") != []:
                failures.append(f"{label} negative_control gold must be empty lists")
        entities = gold.get("entities", [])
        relations = gold.get("relations", [])
        if gold_empty:
            # Empty-gold negative controls: validate only that the gold is the
            # canonical empty object. They intentionally do not pass the
            # production typed-evidence schema/evidence check (which would
            # require non-empty evidence substrings) and intentionally do not
            # contribute to type/signature coverage. Their purpose is to test
            # empty-gold false-positive behavior.
            if not isinstance(entities, list) or not isinstance(relations, list):
                failures.append(f"{label} gold_empty requires entities/relations lists")
            if entities != [] or relations != []:
                failures.append(f"{label} gold_empty requires empty entities/relations lists")
            continue
        evaluated = evaluate_response({"condition": "typed_evidence", "context": text}, json.dumps(gold))
        if evaluated["status"] != "valid":
            failures.append(f"{label} gold fails production schema/evidence validation: {evaluated['diagnostics']}")
            continue
        gold_entity_count += len(entities)
        gold_relation_count += len(relations)
        covered_types.update(str(item["type"]) for item in entities)
        covered_signatures.update(_relation_signatures(gold))
        if not domains and any(str(item.get("type")) == "Indicator" and "." in str(item.get("name")) for item in entities):
            # IP indicators are checked separately; this branch is only a defensive
            # reminder that host-like strings must use reserved suffixes.
            for item in entities:
                name = str(item.get("name", ""))
                if str(item.get("type")) == "Indicator" and re.search(r"[A-Za-z]", name):
                    if not name.endswith((".example", ".invalid")):
                        failures.append(f"{label} uses a non-reserved indicator domain")

    if negative_control_count != EXPECTED_NEGATIVE_CONTROLS:
        failures.append(
            f"exactly {EXPECTED_NEGATIVE_CONTROLS} empty-gold negative controls are required (found {negative_control_count})"
        )

    if len(ids) != len(set(ids)):
        failures.append("context IDs are not unique")
    if len(texts) != len(set(texts)):
        failures.append("context texts are not unique")
    if len(context_hashes) != len(set(context_hashes)):
        failures.append("context text hashes are not unique")
    if covered_types != ENTITY_TYPES:
        failures.append(f"entity type coverage differs: {sorted(covered_types)}")
    if covered_signatures != RELATION_SIGNATURES:
        failures.append(f"relation signature coverage differs: {sorted(covered_signatures)}")
    if failures:
        raise BenchmarkSourceError("; ".join(failures))
    return {
        "context_count": len(contexts),
        "negative_controls": negative_control_count,
        "unique_context_ids": len(set(ids)),
        "unique_context_hashes": len(set(context_hashes)),
        "entity_types": sorted(covered_types),
        "relation_signatures": [list(value) for value in sorted(covered_signatures)],
        "gold_entity_count": gold_entity_count,
        "gold_relation_count": gold_relation_count,
        "reserved_domain_mentions": sum(text.count(".example") + text.count(".invalid") for text in texts),
        "documentation_ipv4_mentions": sum(
            len(IPV4_PATTERN.findall(text)) for text in texts
        ),
    }


def freeze(source_path: Path, output_dir: Path) -> dict[str, Any]:
    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise BenchmarkSourceError("author source must be a JSON object")
    audit = validate_source(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts: list[dict[str, Any]] = []
    for ordinal, row in enumerate(source["contexts"], start=1):
        context = dict(row)
        context["ordinal"] = ordinal
        context["sample_id"] = context["context_id"]
        context["group"] = context["family"]
        context["source_path"] = source_path.relative_to(PAPER_ROOT).as_posix()
        context["source_row_index"] = ordinal - 1
        context["source_sha256"] = sha256_file(source_path)
        context["context_sha256"] = stable_hash(context["text"])
        context["gold_sha256"] = stable_hash(context["gold"])
        contexts.append(context)
    contexts_path = output_dir / "contexts.jsonl"
    write_jsonl(contexts_path, contexts)

    dedication_path = output_dir / "CC0-1.0-DEDICATION.md"
    dedication_path.write_text(
        "# CC0-1.0 dedication\n\n"
        "The authors waive, to the fullest extent permitted by law, copyright and related rights "
        "in `author_source.json`, `contexts.jsonl`, the fictitious benchmark prose, the gold annotations, "
        "and the simulated injected-note text. These materials are dedicated to the public domain under "
        "Creative Commons CC0 1.0 Universal (SPDX: CC0-1.0). No third-party CTI text is included.\n\n"
        "This dedication does not apply to software files, model responses, or third-party dependencies.\n",
        encoding="utf-8",
    )
    family_counts = Counter(str(row["family"]) for row in contexts)
    ledger_path = output_dir / "DESIGN_LEDGER.md"
    ledger_path.write_text(
        f"""# Round-3 CC0 controlled benchmark design ledger

## Frozen design

- Benchmark: `{source['benchmark_id']}`
- Contexts: {audit['context_count']} original author-written synthetic texts
  ({audit['negative_controls']} of them empty-gold negative controls)
- Gold: {audit['gold_entity_count']} entities and {audit['gold_relation_count']} relations, frozen before calls
- Coverage: all five supported entity types and all four directed relation signatures
  (coverage is required only on the 14 non-negative-control contexts)
- Families: `{json.dumps(dict(sorted(family_counts.items())), sort_keys=True)}`
- Conditions: minimal JSON; typed evidence; typed evidence plus an original simulated untrusted note
- Repeats: 3 per context/condition (144 hosted calls total)
- Order: deterministic Latin rotation; each condition occupies each within-context ordinal once across repeats
- Temperature: 0.0
- Gold policy: the injected note never changes the clean context or its gold graph
- Protocol version: `{source.get('protocol_version', 'unspecified')}`
- Pilot-driven pre-confirmatory amendment: {source.get('pilot_amendment', 'none')}

## Analysis unit

The 16 context clusters are the inferential/descriptive unit of analysis. The
three prompt conditions and three repeats are nested within each cluster, so
the 144 requests are NOT 144 independent samples. All gold-score and paired
displacement reporting must use per-cluster nested summaries, not treat the
144 requests as independent observations.

## Empty-gold negative controls

The two empty-gold negative controls (`r3c15`, `r3c16`) carry gold
`{{"entities": [], "relations": []}}` and are marked `negative_control: true`
and `gold_empty: true`. They test empty-gold false-positive behavior: a model
that emits non-empty entities/relations when the gold is empty registers as
false positives. They are excluded from the entity-type / relation-signature
coverage requirement.

## Construct boundary

Exact entity/relation set metrics measure performance only on this deliberately
authored controlled design set. They are not estimates of performance on real
CTI, do not define a population sampling frame, and do not license claims of
generalization, deployment readiness, or operational robustness.

## Governance rubric

1. No context was copied or paraphrased from third-party CTI.
2. Names are fictitious; IPv4/domain strings use documentation reservations;
   CVE-like identifiers use the fictitious year 2099.
3. No personal, sensitive, credential, customer, or live-incident data occur.
4. Every gold evidence string is a non-empty exact substring of its clean context
   (the empty-gold negative controls have no gold evidence).
5. Gold passes the production extraction schema and covers all supported
   types/signatures (the 14 non-negative-control contexts).
6. Source, context, and gold hashes are frozen before hosted calls.
7. CC0 applies to the authored dataset; hosted model outputs remain separate audit data.

## Reproduction

```bash
python src/generate_round3_cc0_benchmark.py \\
  --source benchmark/round3_cc0/author_source.json \\
  --output-dir benchmark/round3_cc0
```
""",
        encoding="utf-8",
    )

    generated_at = utc_now()
    g0 = {
        "gate_id": "G0_round3_cc0_data_provenance",
        "stage": "dataset_acquisition",
        "hypothesis_or_mechanism": "The benchmark is original, explicitly dedicated, non-sensitive, and hash-addressed before hosted calls.",
        "input_artifacts": [str(source_path), str(contexts_path), str(dedication_path)],
        "command_or_check": "generate_round3_cc0_benchmark.py deterministic source validation",
        "pass_criteria": {"license": "CC0-1.0", "contexts": EXPECTED_CONTEXT_COUNT, "negative_controls": EXPECTED_NEGATIVE_CONTROLS, "unique_ids_and_hashes": EXPECTED_CONTEXT_COUNT, "third_party_cti_text": 0, "personal_or_sensitive_data": 0},
        "metrics": {**audit, "license": source["license"], "authorship_statement_present": True, "privacy_statement_present": True},
        "failed_criteria": [],
        "verdict": "PASS",
        "fail_interpretation": "Hosted execution would lack a governance-clean, traceable input benchmark.",
        "next_allowed_step": "case schedule generation and G1 integrity validation",
    }
    write_json(output_dir / "gate_G0_data_provenance.json", g0)

    manifest = {
        "schema_version": "1.0",
        "benchmark_id": source["benchmark_id"],
        "generated_at": generated_at,
        "generator": Path(__file__).relative_to(PAPER_ROOT).as_posix(),
        "generator_sha256": sha256_file(Path(__file__)),
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (source_path, contexts_path, dedication_path, ledger_path, output_dir / "gate_G0_data_provenance.json")
        },
        "audit": audit,
    }
    write_json(output_dir / "benchmark_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = freeze(args.source, args.output_dir)
    print(json.dumps({"benchmark_id": manifest["benchmark_id"], "audit": manifest["audit"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fail-closed OWL/SWRL validation and reasoning for AutoMalKG."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from owlready2 import (
    AllDisjoint,
    Imp,
    ObjectProperty,
    OwlReadyInconsistentOntologyError,
    Thing,
    World,
    sync_reasoner_pellet,
)

try:  # Support script and package-style execution.
    from .contracts import ReasoningResult, ReasoningStatus
    from .extraction import (
        RELATION_SIGNATURES,
        relation_key,
        validate_extraction_schema,
    )
except ImportError:  # pragma: no cover - exercised by script entry points.
    from contracts import ReasoningResult, ReasoningStatus
    from extraction import (
        RELATION_SIGNATURES,
        relation_key,
        validate_extraction_schema,
    )


ACTOR_TARGET_RULE_ID = "swrl_actor_uses_malware_targets_vulnerability_v1"


def _owl_local_name(entity_id: str) -> str:
    """Map an external ID to a stable, collision-resistant OWL local name."""

    slug = re.sub(r"[^A-Za-z0-9_]", "_", entity_id).strip("_") or "entity"
    digest = sha256(entity_id.encode("utf-8")).hexdigest()[:12]
    return f"entity_{slug[:48]}_{digest}"


class MalOntReasoner:
    """Populate an isolated MALOnt world and invoke Pellet fail-closed."""

    def __init__(self, ontology_iri: str = "http://example.org/malont.owl#") -> None:
        self.world = World()
        self.onto = self.world.get_ontology(ontology_iri)
        self.entity_map = {}
        self.entity_records = {}
        self.entity_types = {}
        self.owl_name_to_entity_id = {}
        self.asserted_relations = {}
        self._asserted_relation_count = 0
        self._load_result = ReasoningResult(ReasoningStatus.NOT_RUN)
        self.build_ontology()

    def build_ontology(self) -> None:
        """Build the minimal class/property vocabulary and one SWRL rule."""

        with self.onto:
            class Entity(Thing):
                pass

            class ThreatActor(Entity):
                pass

            class Malware(Entity):
                pass

            class Vulnerability(Entity):
                pass

            class Indicator(Entity):
                pass

            class Campaign(Entity):
                pass

            AllDisjoint([ThreatActor, Malware, Vulnerability, Indicator, Campaign])

            class Uses(ObjectProperty):
                domain = [ThreatActor]
                range = [Malware]

            class Targets(ObjectProperty):
                # Entity is intentional: the SWRL conclusion targets actors,
                # while asserted extraction Targets edges originate in Malware.
                domain = [Entity]
                range = [Vulnerability]

            class AssociatedWith(ObjectProperty):
                domain = [Malware]
                range = [Campaign]

            class HasIndicator(ObjectProperty):
                domain = [Malware]
                range = [Indicator]

            rule = Imp()
            rule.set_as_rule(
                "ThreatActor(?a), Uses(?a, ?m), Malware(?m), "
                "Targets(?m, ?v) -> Targets(?a, ?v)"
            )

    @staticmethod
    def _read_payload(filepath: Any) -> Tuple[Optional[Mapping[str, Any]], list]:
        path = Path(filepath)
        if not path.exists():
            return None, [f"extraction file not found: {path}"]
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            return None, [f"cannot read extraction file: {type(exc).__name__}: {exc}"]
        if not isinstance(data, Mapping):
            return None, ["extraction file root must be a JSON object"]
        return data, []

    def load_extracted_json(self, filepath: Any) -> ReasoningResult:
        """Read, validate, then atomically populate extraction facts."""

        data, diagnostics = self._read_payload(filepath)
        if data is None:
            self._load_result = ReasoningResult(
                ReasoningStatus.VALIDATION_ERROR, diagnostics=diagnostics
            )
            return self._load_result
        return self.load_extracted_data(data)

    def load_extracted_data(self, data: Mapping[str, Any]) -> ReasoningResult:
        """Validate all IDs, predicates, endpoints, and types before mutation."""

        serialized_status = data.get("status")
        if serialized_status is not None and serialized_status != "success":
            self._load_result = ReasoningResult(
                ReasoningStatus.VALIDATION_ERROR,
                diagnostics=[
                    f"extraction status must be 'success', got {serialized_status!r}"
                ],
            )
            return self._load_result

        entities, relations, diagnostics = validate_extraction_schema(
            data, allow_result_envelope=True
        )
        if diagnostics:
            self._load_result = ReasoningResult(
                ReasoningStatus.VALIDATION_ERROR, diagnostics=diagnostics
            )
            return self._load_result
        if not entities:
            self._load_result = ReasoningResult(
                ReasoningStatus.VALIDATION_ERROR,
                diagnostics=["reasoning requires at least one validated entity"],
            )
            return self._load_result
        if self.entity_map:
            self._load_result = ReasoningResult(
                ReasoningStatus.VALIDATION_ERROR,
                diagnostics=["reasoner instance is already populated"],
            )
            return self._load_result

        # Mutation starts only after the full payload has passed validation.
        with self.onto:
            for entity in entities:
                local_name = _owl_local_name(entity["id"])
                owl_class = getattr(self.onto, entity["type"])
                individual = owl_class(local_name)
                individual.label = [entity["name"]]
                self.entity_map[entity["id"]] = individual
                self.entity_records[entity["id"]] = dict(entity)
                self.entity_types[entity["id"]] = entity["type"]
                self.owl_name_to_entity_id[individual.name] = entity["id"]

            for relation in relations:
                subject = self.entity_map[relation["subject"]]
                obj = self.entity_map[relation["object"]]
                getattr(subject, relation["predicate"]).append(obj)
                key = (
                    relation["subject"],
                    relation["predicate"],
                    relation["object"],
                )
                self.asserted_relations[key] = dict(relation)

        self._asserted_relation_count = len(relations)
        self._load_result = ReasoningResult(
            ReasoningStatus.VALIDATED,
            entity_count=len(entities),
            relation_count=len(relations),
        )
        return self._load_result

    def _snapshot_relations(self) -> set:
        """Return all current supported property assertions as external IDs."""

        relations = set()
        for subject_id, subject in self.entity_map.items():
            for predicate in sorted(RELATION_SIGNATURES):
                for obj in getattr(subject, predicate, ()):  # inferred included
                    object_id = self.owl_name_to_entity_id.get(obj.name)
                    if object_id is not None:
                        relations.add((subject_id, predicate, object_id))
        return relations

    def _proofs_for_actor_target(self, subject_id: str, object_id: str) -> list:
        """Return every asserted two-edge proof for the configured SWRL rule."""

        if self.entity_types.get(subject_id) != "ThreatActor":
            return []
        if self.entity_types.get(object_id) != "Vulnerability":
            return []

        proofs = []
        malware_ids = sorted(
            obj
            for subject, predicate, obj in self.asserted_relations
            if subject == subject_id
            and predicate == "Uses"
            and self.entity_types.get(obj) == "Malware"
        )
        for malware_id in malware_ids:
            first_key = (subject_id, "Uses", malware_id)
            second_key = (malware_id, "Targets", object_id)
            if second_key not in self.asserted_relations:
                continue
            supports = [
                self.asserted_relations[first_key],
                self.asserted_relations[second_key],
            ]
            proofs.append(
                {
                    "support_relation_ids": [support["id"] for support in supports],
                    "support_relation_keys": [support["key"] for support in supports],
                    "provenance": sorted(
                        {
                            provenance
                            for support in supports
                            for provenance in support.get("provenance", ())
                        }
                    ),
                }
            )
        return proofs

    @staticmethod
    def _reasoner_unavailable(exc: Exception) -> bool:
        message = f"{type(exc).__name__}: {exc}".casefold()
        return isinstance(exc, (FileNotFoundError, PermissionError)) or any(
            marker in message
            for marker in ("java", "pellet", "could not find or load main class")
        )

    def verify_and_infer(self) -> ReasoningResult:
        """Run Pellet; unavailable or failed reasoning never counts as success."""

        if self._load_result.status is not ReasoningStatus.VALIDATED:
            return ReasoningResult(
                ReasoningStatus.NOT_RUN,
                diagnostics=[
                    "reasoning blocked because extraction was not successfully validated"
                ],
                entity_count=len(self.entity_map),
                relation_count=self._asserted_relation_count,
            )

        before = self._snapshot_relations()
        try:
            with self.onto:
                sync_reasoner_pellet(
                    [self.onto],
                    infer_property_values=True,
                    infer_data_property_values=True,
                )
        except OwlReadyInconsistentOntologyError as exc:
            return ReasoningResult(
                ReasoningStatus.INCONSISTENT,
                diagnostics=[f"Pellet reported inconsistency: {exc}"],
                entity_count=len(self.entity_map),
                relation_count=len(before),
            )
        except Exception as exc:  # Fail closed for every execution failure.
            status = (
                ReasoningStatus.REASONER_UNAVAILABLE
                if self._reasoner_unavailable(exc)
                else ReasoningStatus.REASONER_ERROR
            )
            return ReasoningResult(
                status,
                diagnostics=[f"Pellet execution failed: {type(exc).__name__}: {exc}"],
                entity_count=len(self.entity_map),
                relation_count=len(before),
            )

        after = self._snapshot_relations()
        inferred = []
        diagnostics = []
        for subject, predicate, obj in sorted(after - before):
            record = {"subject": subject, "predicate": predicate, "object": obj}
            if predicate == "Targets":
                proofs = self._proofs_for_actor_target(subject, obj)
                if proofs:
                    record.update(
                        rule_id=ACTOR_TARGET_RULE_ID,
                        proofs=proofs,
                        provenance=sorted(
                            {
                                provenance
                                for proof in proofs
                                for provenance in proof["provenance"]
                            }
                        ),
                    )
            if "proofs" not in record:
                diagnostics.append(
                    "no asserted support ledger for inferred relation: "
                    f"{relation_key(subject, predicate, obj)}"
                )
            inferred.append(record)
        return ReasoningResult(
            ReasoningStatus.CONSISTENT,
            diagnostics=diagnostics,
            inferred_relations=inferred,
            entity_count=len(self.entity_map),
            relation_count=len(after),
        )

    def serialize(self, filepath: Any, result: ReasoningResult) -> Path:
        """Serialize only a graph proven consistent by a completed reasoner run."""

        if result.status is not ReasoningStatus.CONSISTENT:
            raise ValueError(
                f"refusing to serialize graph with reasoning status {result.status.value}"
            )
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.onto.save(file=str(path), format="rdfxml")
        return path


if __name__ == "__main__":  # pragma: no cover - manual entry point.
    reasoner = MalOntReasoner()
    loaded = reasoner.load_extracted_json("../data/processed/extracted_knowledge.json")
    result = reasoner.verify_and_infer()
    print(json.dumps({"load": loaded, "reasoning": result}, indent=2))
    if result.status is ReasoningStatus.CONSISTENT:
        reasoner.serialize("../data/processed/malont_populated.owl", result)

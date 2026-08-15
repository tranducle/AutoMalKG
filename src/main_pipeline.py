"""Fail-closed orchestration for extraction, reasoning, and serialization."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

try:  # Support script and package-style execution.
    from .contracts import ExtractionStatus, ReasoningStatus
    from .extraction import MalOntExtractor
    from .reasoning import MalOntReasoner
except ImportError:  # pragma: no cover - exercised by script entry points.
    from contracts import ExtractionStatus, ReasoningStatus
    from extraction import MalOntExtractor
    from reasoning import MalOntReasoner


async def run_pipeline(
    chunks: Sequence[Any],
    *,
    output_dir: Optional[Any] = None,
    extractor: Optional[MalOntExtractor] = None,
    reasoner: Optional[MalOntReasoner] = None,
) -> dict:
    """Run the pipeline and serialize outputs only after ``CONSISTENT``."""

    extractor = extractor or MalOntExtractor()
    reasoner = reasoner or MalOntReasoner()
    extraction = await extractor.process_document_chunks(chunks)
    outcome = {
        "status": "blocked",
        "serialized": False,
        "extraction": extraction,
        "load": None,
        "reasoning": None,
        "artifacts": {},
    }
    if extraction.status is not ExtractionStatus.SUCCESS:
        outcome["status"] = "extraction_failed"
        return outcome

    load_result = reasoner.load_extracted_data(extraction)
    outcome["load"] = load_result
    if load_result.status is not ReasoningStatus.VALIDATED:
        outcome["status"] = "validation_failed"
        return outcome

    reasoning = reasoner.verify_and_infer()
    outcome["reasoning"] = reasoning
    if reasoning.status is not ReasoningStatus.CONSISTENT:
        outcome["status"] = "reasoning_failed"
        return outcome

    target_dir = Path(output_dir) if output_dir is not None else (
        Path(__file__).resolve().parent.parent / "data" / "processed"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    extraction_path = target_dir / "extracted_knowledge.json"
    ontology_path = target_dir / "malont_populated.owl"

    # Stage both files and atomically promote them only after both serialize.
    with tempfile.TemporaryDirectory(prefix="automalkg_", dir=target_dir) as temp_dir:
        staged_extraction = Path(temp_dir) / extraction_path.name
        staged_ontology = Path(temp_dir) / ontology_path.name
        with staged_extraction.open("w", encoding="utf-8") as handle:
            json.dump(extraction, handle, indent=2, sort_keys=True)
            handle.write("\n")
        reasoner.serialize(staged_ontology, reasoning)
        os.replace(staged_extraction, extraction_path)
        os.replace(staged_ontology, ontology_path)

    outcome.update(
        status="success",
        serialized=True,
        artifacts={
            "extraction": str(extraction_path),
            "ontology": str(ontology_path),
        },
    )
    return outcome


async def main() -> None:
    """Run a small local smoke case using configured LLM providers."""

    sample_texts = [
        "APT29 uses the CozyCar malware, which targets CVE-2021-1234.",
        "AppleJeus is associated with a Lazarus campaign.",
    ]
    result = await run_pipeline(sample_texts)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover - manual entry point.
    asyncio.run(main())

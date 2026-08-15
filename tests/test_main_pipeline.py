"""P0 integration tests for the orchestrator's serialization gate."""

import asyncio
from pathlib import Path

from contracts import (
    ExtractionResult,
    ExtractionStatus,
    ReasoningResult,
    ReasoningStatus,
)
from main_pipeline import run_pipeline


class StubExtractor:
    async def process_document_chunks(self, _chunks):
        return ExtractionResult(
            ExtractionStatus.SUCCESS,
            entities=[{"id": "m", "type": "Malware", "name": "M"}],
        )


class StubReasoner:
    def __init__(self, final_status):
        self.final_status = final_status
        self.serialize_calls = 0

    def load_extracted_data(self, _data):
        return ReasoningResult(
            ReasoningStatus.VALIDATED, entity_count=1, relation_count=0
        )

    def verify_and_infer(self):
        return ReasoningResult(self.final_status, entity_count=1, relation_count=0)

    def serialize(self, filepath, _result):
        self.serialize_calls += 1
        Path(filepath).write_text("<owl/>", encoding="utf-8")


def test_non_consistent_result_blocks_every_serialized_artifact(tmp_path):
    reasoner = StubReasoner(ReasoningStatus.REASONER_UNAVAILABLE)
    result = asyncio.run(
        run_pipeline(
            ["text"], output_dir=tmp_path, extractor=StubExtractor(), reasoner=reasoner
        )
    )

    assert result["status"] == "reasoning_failed"
    assert result["serialized"] is False
    assert reasoner.serialize_calls == 0
    assert list(tmp_path.iterdir()) == []


def test_consistent_result_promotes_both_staged_artifacts(tmp_path):
    reasoner = StubReasoner(ReasoningStatus.CONSISTENT)
    result = asyncio.run(
        run_pipeline(
            ["text"], output_dir=tmp_path, extractor=StubExtractor(), reasoner=reasoner
        )
    )

    assert result["status"] == "success"
    assert result["serialized"] is True
    assert reasoner.serialize_calls == 1
    assert (tmp_path / "extracted_knowledge.json").exists()
    assert (tmp_path / "malont_populated.owl").exists()

"""P0 tests for deterministic, structured ingestion chunks."""

import pytest
from pathlib import Path

from data_ingestion import DataIngestionPipeline


def test_default_directories_are_independent_of_caller_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pipeline = DataIngestionPipeline()
    project_root = Path(__file__).resolve().parents[1]

    assert pipeline.raw_dir == project_root / "data" / "raw"
    assert pipeline.processed_dir == project_root / "data" / "processed"


def test_explicit_relative_directories_remain_caller_controlled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pipeline = DataIngestionPipeline("custom-raw", "custom-processed")

    assert pipeline.raw_dir == Path("custom-raw")
    assert pipeline.processed_dir == Path("custom-processed")


def test_structured_chunks_are_stable_and_record_offsets_and_hashes():
    text = "zero one two three four five"
    first = DataIngestionPipeline.sliding_window_chunker(
        text, chunk_size=4, overlap=2, source_id="sample.pdf"
    )
    second = DataIngestionPipeline.sliding_window_chunker(
        text, chunk_size=4, overlap=2, source_id="sample.pdf"
    )

    assert first == second
    assert [(item.start_word, item.end_word) for item in first] == [(0, 4), (2, 6)]
    assert all(item.chunk_id.startswith("chunk_") for item in first)
    assert all(len(item.text_sha256) == 64 for item in first)


@pytest.mark.parametrize(
    "chunk_size, overlap", [(0, 0), (4, -1), (4, 4), (4, 5)]
)
def test_chunker_rejects_parameters_that_can_stall_or_corrupt_offsets(
    chunk_size, overlap
):
    with pytest.raises(ValueError):
        DataIngestionPipeline.sliding_window_chunker(
            "some words", chunk_size=chunk_size, overlap=overlap
        )

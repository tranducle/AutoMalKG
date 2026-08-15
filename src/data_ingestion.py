"""Deterministic PDF ingestion and content-addressed text chunking."""

from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional

try:  # Support script and package-style execution.
    from .contracts import StructuredChunk
except ImportError:  # pragma: no cover - exercised by script entry points.
    from contracts import StructuredChunk


class DataIngestionPipeline:
    """Extract PDFs and emit structured chunks with stable provenance."""

    def __init__(
        self,
        raw_dir: Optional[str] = None,
        processed_dir: Optional[str] = None,
    ) -> None:
        project_root = Path(__file__).resolve().parent.parent
        self.raw_dir = Path(raw_dir) if raw_dir is not None else project_root / "data" / "raw"
        self.processed_dir = (
            Path(processed_dir)
            if processed_dir is not None
            else project_root / "data" / "processed"
        )
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from a PDF; parsing errors are explicit to callers."""

        # Lazy import isolates the optional native dependency from pure chunking
        # workflows and avoids loading an incompatible binary at module import.
        import fitz  # PyMuPDF

        pages = []
        with fitz.open(pdf_path) as document:
            for page in document:
                pages.append(page.get_text())
        return "\n".join(pages)

    @staticmethod
    def date_filter_post_may_2024(metadata_date) -> bool:
        """Return whether a parseable report date is later than 2024-05-01."""

        cutoff = datetime(2024, 5, 1)
        if isinstance(metadata_date, str):
            try:
                metadata_date = datetime.strptime(metadata_date, "%Y-%m-%d")
            except ValueError:
                return False
        if not isinstance(metadata_date, datetime):
            return False
        return metadata_date > cutoff

    @staticmethod
    def sliding_window_chunker(
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
        *,
        source_id: str = "inline",
        document_sha256: Optional[str] = None,
    ) -> List[StructuredChunk]:
        """Split text into overlapping, content-addressed structured chunks."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

        words = text.split()
        if not words:
            return []
        document_digest = document_sha256 or sha256(text.encode("utf-8")).hexdigest()
        chunks = []
        start = 0
        index = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                StructuredChunk.create(
                    text=chunk_text,
                    source_id=source_id,
                    index=index,
                    start_word=start,
                    end_word=end,
                    document_sha256=document_digest,
                )
            )
            if end == len(words):
                break
            start += chunk_size - overlap
            index += 1
        return chunks

    def process_all_reports(self) -> Dict[str, List[StructuredChunk]]:
        """Extract every PDF recursively and persist one JSONL chunk manifest."""

        pdf_files = sorted(self.raw_dir.rglob("*.pdf"))
        processed_data = {}
        for pdf_path in pdf_files:
            source_id = pdf_path.relative_to(self.raw_dir).as_posix()
            text = self.extract_text_from_pdf(str(pdf_path))
            document_digest = sha256(pdf_path.read_bytes()).hexdigest()
            chunks = self.sliding_window_chunker(
                text,
                source_id=source_id,
                document_sha256=document_digest,
            )
            processed_data[source_id] = chunks

            safe_name = sha256(source_id.encode("utf-8")).hexdigest()[:20]
            output_path = self.processed_dir / f"{safe_name}.chunks.jsonl"
            with output_path.open("w", encoding="utf-8") as handle:
                for chunk in chunks:
                    handle.write(json.dumps(chunk.to_dict(), sort_keys=True) + "\n")
        return processed_data


if __name__ == "__main__":  # pragma: no cover - manual entry point.
    pipeline = DataIngestionPipeline()
    result = pipeline.process_all_reports()
    print(f"Processed {len(result)} reports.")

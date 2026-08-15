"""Strict baseline loaders for reproducible named checkpoints.

Optional heavyweight dependencies are imported only when a baseline is
requested.  Missing packages, models, or checkpoints raise typed failures;
they never degrade to empty predictions that could be scored as a real run.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional


class BaselineUnavailableError(RuntimeError):
    """Raised when a requested baseline cannot produce valid predictions."""


class BaselineModels:
    """Load explicitly named spaCy and Hugging Face token-classification models."""

    GENERIC_NER_CHECKPOINTS = frozenset(
        {
            "dslim/bert-base-NER",
            "dbmdz/bert-large-cased-finetuned-conll03-english",
        }
    )

    def __init__(self) -> None:
        self.spacy_model: Optional[Any] = None
        self.roberta_pipeline: Optional[Any] = None
        self.spacy_model_name: Optional[str] = None
        self.roberta_model_id: Optional[str] = None

    def load_spacy(self, model_name: str = "en_core_web_trf") -> Any:
        """Load a named spaCy model or fail visibly."""

        try:
            spacy = importlib.import_module("spacy")
        except ImportError as exc:
            raise BaselineUnavailableError(
                "spaCy is not installed; install the pinned project requirements"
            ) from exc
        try:
            model = spacy.load(model_name)
        except (OSError, ImportError) as exc:
            raise BaselineUnavailableError(
                f"spaCy checkpoint is unavailable: {model_name}"
            ) from exc
        self.spacy_model = model
        self.spacy_model_name = model_name
        return model

    def load_roberta(self, model_id: str) -> Any:
        """Load an explicitly selected domain checkpoint or fail visibly.

        Generic CoNLL NER checkpoints are intentionally rejected because they
        are not a Cyber-RoBERTa baseline and must not be relabeled as one.
        """

        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must name an explicit Hugging Face checkpoint")
        model_id = model_id.strip()
        if model_id in self.GENERIC_NER_CHECKPOINTS:
            raise ValueError(
                f"generic NER checkpoint cannot stand in for Cyber-RoBERTa: {model_id}"
            )
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise BaselineUnavailableError(
                "transformers is not installed; install the pinned project requirements"
            ) from exc
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
            model = transformers.AutoModelForTokenClassification.from_pretrained(
                model_id
            )
            ner_pipeline = transformers.pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )
        except Exception as exc:
            raise BaselineUnavailableError(
                f"Hugging Face checkpoint is unavailable or incompatible: {model_id}"
            ) from exc
        self.roberta_pipeline = ner_pipeline
        self.roberta_model_id = model_id
        return ner_pipeline

    def extract_spacy(self, text: str) -> list[tuple[str, str]]:
        if self.spacy_model is None:
            raise BaselineUnavailableError("spaCy baseline has not been loaded")
        doc = self.spacy_model(text)
        return [(entity.text, entity.label_) for entity in doc.ents]

    def extract_roberta(self, text: str) -> Any:
        if self.roberta_pipeline is None:
            raise BaselineUnavailableError("Hugging Face baseline has not been loaded")
        return self.roberta_pipeline(text)


if __name__ == "__main__":  # pragma: no cover - no implicit downloads.
    raise SystemExit(
        "No baseline is selected. Import BaselineModels and pass explicit checkpoint IDs."
    )

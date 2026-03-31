from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


SentimentLabel = Literal["negative", "neutral", "positive"]


@dataclass(slots=True)
class Signal:
    source: str
    label: SentimentLabel
    confidence: float
    score: float
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "score": round(self.score, 4),
            "details": self.details,
        }


@dataclass(slots=True)
class RetrievedExample:
    text: str
    label: SentimentLabel
    similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "similarity": round(self.similarity, 4),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class PredictionResult:
    text: str
    normalized_text: str
    label: SentimentLabel
    confidence: float
    signals: list[Signal]
    retrieved_examples: list[RetrievedExample]
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "signals": [signal.as_dict() for signal in self.signals],
            "retrieved_examples": [example.as_dict() for example in self.retrieved_examples],
            "rationale": self.rationale,
            "metadata": self.metadata,
        }


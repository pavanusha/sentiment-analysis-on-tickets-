from __future__ import annotations

from .scoring import clamp, label_to_score, score_to_label
from .types import Signal


class TransformerSentimentModel:
    def __init__(
        self,
        model_name: str,
        max_text_length: int = 512,
        enabled: bool = True,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_text_length = max_text_length
        self.enabled = enabled
        self.local_files_only = local_files_only
        self._pipeline = None
        self._load_attempted = False

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._pipeline is not None

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if not self.enabled:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            load_kwargs = {}
            if self.local_files_only:
                load_kwargs["local_files_only"] = True

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_kwargs)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **load_kwargs,
            )

            self._pipeline = pipeline(
                task="text-classification",
                model=model,
                tokenizer=tokenizer,
            )
        except Exception:
            self._pipeline = None

    def _normalize_label(self, label: str) -> str:
        normalized = label.strip().lower()
        mapping = {
            "label_0": "negative",
            "label_1": "neutral",
            "label_2": "positive",
            "negative": "negative",
            "neutral": "neutral",
            "positive": "positive",
        }
        if normalized in mapping:
            return mapping[normalized]
        if "neg" in normalized:
            return "negative"
        if "neu" in normalized:
            return "neutral"
        if "pos" in normalized:
            return "positive"
        return "neutral"

    def predict(self, text: str) -> Signal | None:
        self._ensure_loaded()
        if self._pipeline is None:
            return None

        try:
            prediction = self._pipeline(
                text,
                truncation=True,
                max_length=self.max_text_length,
            )
        except Exception:
            return None

        if isinstance(prediction, list):
            prediction = prediction[0]

        label = self._normalize_label(str(prediction.get("label", "neutral")))
        confidence = clamp(float(prediction.get("score", 0.5)))
        score = 0.0 if label == "neutral" else label_to_score(label, confidence)

        return Signal(
            source="transformer",
            label=label,
            confidence=confidence,
            score=score,
            details={"model": self.model_name},
        )


class VaderSentimentModel:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._analyzer = None
        self._load_attempted = False

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._analyzer is not None

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if not self.enabled:
            return
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._analyzer = SentimentIntensityAnalyzer()
        except Exception:
            self._analyzer = None

    def predict(self, text: str) -> Signal | None:
        self._ensure_loaded()
        if self._analyzer is None:
            return None

        try:
            scores = self._analyzer.polarity_scores(text)
        except Exception:
            return None

        compound = float(scores.get("compound", 0.0))
        label = score_to_label(compound, neutral_window=0.16)
        confidence = clamp(0.45 + (0.45 * abs(compound)), 0.0, 0.92)

        return Signal(
            source="vader",
            label=label,
            confidence=confidence,
            score=compound,
            details=scores,
        )

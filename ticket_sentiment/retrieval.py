from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .preprocessing import normalize_text, tokenize
from .scoring import clamp, label_to_score, score_to_label
from .types import RetrievedExample, Signal, SentimentLabel

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


@dataclass(slots=True)
class ReferenceTicket:
    text: str
    label: SentimentLabel
    metadata: dict[str, object]
    normalized_text: str


class ReferenceTicketStore:
    def __init__(
        self,
        reference_path: Path,
        embedding_model_name: str,
        enable_embeddings: bool = True,
        local_files_only: bool = False,
    ) -> None:
        self.reference_path = reference_path
        self.embedding_model_name = embedding_model_name
        self.enable_embeddings = enable_embeddings
        self.local_files_only = local_files_only
        self.examples = self._load_examples(reference_path)
        self.backend = "token-overlap"
        self._embedding_model = None
        self._embeddings = None
        self._vectorizer = None
        self._matrix = None
        self._token_sets: list[set[str]] = []
        self._prepare_backend()

    def _load_examples(self, reference_path: Path) -> list[ReferenceTicket]:
        if not reference_path.exists():
            raise FileNotFoundError(
                f"Reference ticket file not found: {reference_path}"
            )

        examples: list[ReferenceTicket] = []
        with reference_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                text = str(payload["text"]).strip()
                label = str(payload["label"]).strip().lower()
                if label not in {"negative", "neutral", "positive"}:
                    raise ValueError(f"Invalid label in {reference_path}: {label}")
                metadata = {
                    key: value
                    for key, value in payload.items()
                    if key not in {"text", "label"}
                }
                examples.append(
                    ReferenceTicket(
                        text=text,
                        label=label,
                        metadata=metadata,
                        normalized_text=normalize_text(text),
                    )
                )
        return examples

    def _prepare_backend(self) -> None:
        normalized_texts = [example.normalized_text for example in self.examples]

        if self.enable_embeddings and np is not None:
            try:
                from sentence_transformers import SentenceTransformer

                init_kwargs = {}
                if self.local_files_only:
                    init_kwargs["local_files_only"] = True
                self._embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    **init_kwargs,
                )
                self._embeddings = self._embedding_model.encode(
                    normalized_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                self.backend = "sentence-transformers"
                return
            except Exception:
                self._embedding_model = None
                self._embeddings = None

        if np is not None:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                self._vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                )
                self._matrix = self._vectorizer.fit_transform(normalized_texts)
                self.backend = "tfidf"
                return
            except Exception:
                self._vectorizer = None
                self._matrix = None

        self._token_sets = [set(tokenize(example.normalized_text)) for example in self.examples]
        self.backend = "token-overlap"

    def retrieve(self, text: str, k: int = 5) -> list[RetrievedExample]:
        normalized_text = normalize_text(text)
        if not normalized_text:
            return []

        scored_examples: list[tuple[ReferenceTicket, float]]
        if self.backend == "sentence-transformers" and self._embedding_model is not None:
            query_embedding = self._embedding_model.encode(
                [normalized_text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            similarities = self._embeddings @ query_embedding
            scored_examples = list(zip(self.examples, similarities.tolist()))
        elif self.backend == "tfidf" and self._vectorizer is not None:
            from sklearn.metrics.pairwise import cosine_similarity

            query_vector = self._vectorizer.transform([normalized_text])
            similarities = cosine_similarity(query_vector, self._matrix)[0]
            scored_examples = list(zip(self.examples, similarities.tolist()))
        else:
            query_tokens = set(tokenize(normalized_text))
            scored_examples = []
            for example, example_tokens in zip(self.examples, self._token_sets):
                if not query_tokens or not example_tokens:
                    similarity = 0.0
                else:
                    overlap = len(query_tokens & example_tokens)
                    union = len(query_tokens | example_tokens)
                    similarity = overlap / union if union else 0.0
                if normalized_text in example.normalized_text or example.normalized_text in normalized_text:
                    similarity += 0.1
                scored_examples.append((example, min(similarity, 1.0)))

        ranked = sorted(scored_examples, key=lambda item: item[1], reverse=True)[:k]
        return [
            RetrievedExample(
                text=example.text,
                label=example.label,
                similarity=float(similarity),
                metadata={**example.metadata, "backend": self.backend},
            )
            for example, similarity in ranked
            if similarity > 0.0
        ]

    def build_signal(self, retrieved_examples: list[RetrievedExample]) -> Signal:
        if not retrieved_examples:
            return Signal(
                source="rag",
                label="neutral",
                confidence=0.2,
                score=0.0,
                details={"backend": self.backend, "votes": {}},
            )

        weights = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        top_similarity = 0.0
        weighted_similarity_sum = 0.0
        top_example: RetrievedExample | None = None

        for example in retrieved_examples:
            similarity = max(example.similarity, 0.01)
            weights[example.label] += similarity
            if example.similarity >= top_similarity:
                top_similarity = example.similarity
                top_example = example
            weighted_similarity_sum += similarity

        if top_example is not None and top_similarity >= 0.92:
            confidence = clamp(0.52 + (0.34 * top_similarity), 0.0, 0.97)
            score = (
                0.0
                if top_example.label == "neutral"
                else label_to_score(top_example.label, max(0.45, confidence))
            )
            return Signal(
                source="rag",
                label=top_example.label,
                confidence=confidence,
                score=score,
                details={
                    "backend": self.backend,
                    "mode": "top-match",
                    "top_similarity": top_similarity,
                    "votes": weights,
                },
            )

        total = sum(weights.values()) or 1.0
        score = (weights["positive"] - weights["negative"]) / total
        neutral_bias = weights["neutral"] / total
        if neutral_bias > 0.42 and abs(score) < 0.24:
            label: SentimentLabel = "neutral"
        else:
            label = score_to_label(score, neutral_window=0.15)

        dominance = max(weights.values()) / total
        avg_similarity = weighted_similarity_sum / len(retrieved_examples)
        confidence = clamp(
            0.25 + (0.35 * dominance) + (0.2 * top_similarity) + (0.15 * avg_similarity),
            0.0,
            0.96,
        )
        if top_similarity < 0.2:
            confidence *= 0.78

        return Signal(
            source="rag",
            label=label,
            confidence=confidence,
            score=score,
            details={"backend": self.backend, "votes": weights},
        )

from __future__ import annotations

from .config import Settings
from .lexicon import lexicon_signal
from .llm import CompositeLLMJudge
from .models import TransformerSentimentModel, VaderSentimentModel
from .preprocessing import normalize_text
from .retrieval import ReferenceTicketStore
from .scoring import aggregate_signals
from .types import PredictionResult, RetrievedExample, Signal


class HybridTicketSentimentAnalyzer:
    def __init__(
        self,
        settings: Settings | None = None,
        retriever: ReferenceTicketStore | None = None,
        transformer: TransformerSentimentModel | None = None,
        vader: VaderSentimentModel | None = None,
        llm_judge: CompositeLLMJudge | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.retriever = retriever or ReferenceTicketStore(
            self.settings.reference_data_path,
            self.settings.embedding_model_name,
            enable_embeddings=self.settings.enable_embedding_retrieval,
            local_files_only=self.settings.local_files_only,
        )
        self.transformer = transformer or TransformerSentimentModel(
            self.settings.transformer_model_name,
            self.settings.max_text_length,
            enabled=self.settings.enable_transformer_model,
            local_files_only=self.settings.local_files_only,
        )
        self.vader = vader or VaderSentimentModel(enabled=self.settings.enable_vader_model)
        self.llm_judge = llm_judge or CompositeLLMJudge(
            provider=self.settings.llm_provider,
            openai_api_key=self.settings.openai_api_key,
            openai_model=self.settings.openai_model,
            ollama_base_url=self.settings.ollama_base_url,
            ollama_model=self.settings.ollama_model,
        )

    def predict(self, text: str) -> PredictionResult:
        if not text or not text.strip():
            raise ValueError("Ticket description cannot be empty.")

        normalized_text = normalize_text(text)
        retrieved_examples = self.retriever.retrieve(
            normalized_text,
            k=self.settings.retrieval_top_k,
        )

        signals: list[Signal] = [
            lexicon_signal(normalized_text, original_text=text),
            self.retriever.build_signal(retrieved_examples),
        ]

        transformer_signal = self.transformer.predict(text)
        if transformer_signal is not None:
            signals.append(transformer_signal)

        vader_signal = self.vader.predict(text)
        if vader_signal is not None:
            signals.append(vader_signal)

        aggregate = aggregate_signals(signals)

        if self._should_consult_llm(normalized_text, signals, aggregate):
            llm_signal = self.llm_judge.predict(
                text=text,
                normalized_text=normalized_text,
                signals=signals,
                retrieved_examples=retrieved_examples,
            )
            if llm_signal is not None:
                signals.append(llm_signal)
                aggregate = aggregate_signals(signals)

        rationale = self._build_rationale(
            label=str(aggregate["label"]),
            signals=signals,
            retrieved_examples=retrieved_examples,
        )

        return PredictionResult(
            text=text,
            normalized_text=normalized_text,
            label=str(aggregate["label"]),
            confidence=float(aggregate["confidence"]),
            signals=signals,
            retrieved_examples=retrieved_examples,
            rationale=rationale,
            metadata={
                "retrieval_backend": self.retriever.backend,
                "llm_used": any(signal.source == "llm" for signal in signals),
                "aggregate": aggregate,
            },
        )

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        return [self.predict(text) for text in texts]

    def _should_consult_llm(
        self,
        normalized_text: str,
        signals: list[Signal],
        aggregate: dict[str, object],
    ) -> bool:
        if not self.llm_judge.available:
            return False
        if self.settings.always_use_llm:
            return True

        labels = {signal.label for signal in signals if signal.confidence >= 0.45}
        mixed_signals = len(labels) >= 2
        has_contrast = any(
            token in normalized_text.split()
            for token in {"but", "however", "though", "although", "still"}
        )
        margin = float(aggregate["margin"])
        confidence = float(aggregate["confidence"])
        return margin < self.settings.llm_margin_threshold or (mixed_signals and has_contrast) or confidence < 0.6

    def _build_rationale(
        self,
        label: str,
        signals: list[Signal],
        retrieved_examples: list[RetrievedExample],
    ) -> str:
        ranked_signals = sorted(
            signals,
            key=lambda signal: signal.confidence,
            reverse=True,
        )[:3]
        signal_summary = ", ".join(
            f"{signal.source}={signal.label}({signal.confidence:.2f})"
            for signal in ranked_signals
        )

        retrieved_summary = ""
        if retrieved_examples:
            retrieved_summary = (
                " Top retrieved examples: "
                + "; ".join(
                    f"{example.label}@{example.similarity:.2f}"
                    for example in retrieved_examples[:3]
                )
                + "."
            )

        return f"Predicted {label} based on {signal_summary}.{retrieved_summary}".strip()

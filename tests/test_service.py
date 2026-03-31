from pathlib import Path
import unittest

from ticket_sentiment.config import Settings
from ticket_sentiment.service import HybridTicketSentimentAnalyzer
from ticket_sentiment.types import RetrievedExample, Signal


class FakeRetriever:
    def __init__(self, label: str, score: float) -> None:
        self.backend = "fake"
        self.label = label
        self.score = score

    def retrieve(self, text: str, k: int = 5) -> list[RetrievedExample]:
        return [
            RetrievedExample(
                text="Similar historical ticket",
                label=self.label,
                similarity=0.91,
                metadata={"backend": self.backend},
            )
        ]

    def build_signal(self, retrieved_examples: list[RetrievedExample]) -> Signal:
        return Signal(
            source="rag",
            label=self.label,
            confidence=0.88,
            score=self.score,
            details={"backend": self.backend},
        )


class FakeModel:
    def __init__(self, signal: Signal | None) -> None:
        self.signal = signal

    def predict(self, text: str) -> Signal | None:
        return self.signal


class FakeLLM:
    def __init__(self, signal: Signal | None, available: bool = True) -> None:
        self.signal = signal
        self.available = available

    def predict(self, text, normalized_text, signals, retrieved_examples):
        return self.signal


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(reference_data_path=Path("data/reference_tickets.jsonl"))
        self.local_settings = Settings(
            reference_data_path=Path("data/reference_tickets.jsonl"),
            enable_embedding_retrieval=False,
            enable_transformer_model=False,
            enable_vader_model=False,
        )

    def test_negative_signal_path(self) -> None:
        analyzer = HybridTicketSentimentAnalyzer(
            settings=self.settings,
            retriever=FakeRetriever("negative", -0.85),
            transformer=FakeModel(
                Signal(
                    source="transformer",
                    label="negative",
                    confidence=0.9,
                    score=-0.9,
                )
            ),
            vader=FakeModel(None),
            llm_judge=FakeLLM(None, available=False),
        )

        result = analyzer.predict("still not working, team is frustrated")
        self.assertEqual(result.label, "negative")
        self.assertGreater(result.confidence, 0.6)

    def test_llm_can_break_ambiguous_tie(self) -> None:
        analyzer = HybridTicketSentimentAnalyzer(
            settings=self.settings,
            retriever=FakeRetriever("neutral", 0.0),
            transformer=FakeModel(
                Signal(
                    source="transformer",
                    label="negative",
                    confidence=0.55,
                    score=-0.55,
                )
            ),
            vader=FakeModel(
                Signal(
                    source="vader",
                    label="positive",
                    confidence=0.52,
                    score=0.52,
                )
            ),
            llm_judge=FakeLLM(
                Signal(
                    source="llm",
                    label="positive",
                    confidence=0.9,
                    score=0.9,
                    details={"rationale": "Confirmed fix and appreciation"},
                ),
                available=True,
            ),
        )

        result = analyzer.predict("it was bad before but thanks, works now")
        self.assertEqual(result.label, "positive")
        self.assertTrue(result.metadata["llm_used"])

    def test_outage_with_emotional_language_is_negative(self) -> None:
        analyzer = HybridTicketSentimentAnalyzer(
            settings=self.local_settings,
            llm_judge=FakeLLM(None, available=False),
        )

        result = analyzer.predict("Server is down again and the team is mad about it.")
        self.assertEqual(result.label, "negative")
        self.assertGreater(result.confidence, 0.7)

    def test_mild_recovery_language_is_positive(self) -> None:
        analyzer = HybridTicketSentimentAnalyzer(
            settings=self.local_settings,
            llm_judge=FakeLLM(None, available=False),
        )

        result = analyzer.predict("This is much better after the update.")
        self.assertEqual(result.label, "positive")

    def test_slow_and_frustrating_ticket_is_negative(self) -> None:
        analyzer = HybridTicketSentimentAnalyzer(
            settings=self.local_settings,
            llm_judge=FakeLLM(None, available=False),
        )

        result = analyzer.predict("The dashboard is slow and frustrating for the team.")
        self.assertEqual(result.label, "negative")

    def test_positive_resolution_overrides_error_wording(self) -> None:
        analyzer = HybridTicketSentimentAnalyzer(
            settings=self.local_settings,
            llm_judge=FakeLLM(None, available=False),
        )

        result = analyzer.predict("Looks good to me, no more errors.")
        self.assertEqual(result.label, "positive")

    def test_exact_neutral_request_stays_neutral(self) -> None:
        analyzer = HybridTicketSentimentAnalyzer(
            settings=self.local_settings,
            llm_judge=FakeLLM(None, available=False),
        )

        result = analyzer.predict("Need help adding one more user to the workspace.")
        self.assertEqual(result.label, "neutral")

    def test_exact_neutral_confirmation_stays_neutral(self) -> None:
        analyzer = HybridTicketSentimentAnalyzer(
            settings=self.local_settings,
            llm_judge=FakeLLM(None, available=False),
        )

        result = analyzer.predict("Could you confirm if the patch was deployed?")
        self.assertEqual(result.label, "neutral")


if __name__ == "__main__":
    unittest.main()

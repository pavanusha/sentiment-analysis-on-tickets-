import unittest

from fastapi.testclient import TestClient

from ticket_sentiment.api import create_app


class FakeResult:
    def __init__(self, text: str) -> None:
        self.text = text

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "label": "positive",
            "confidence": 0.99,
            "rationale": "stubbed analyzer result",
        }


class FakeAnalyzer:
    def predict(self, text: str) -> FakeResult:
        return FakeResult(text)


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app(analyzer_factory=FakeAnalyzer)
        self.client = TestClient(self.app)

    def test_root_returns_browser_form(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn('id="predict-form"', response.text)
        self.assertIn("Ticket Sentiment Analyzer", response.text)

    def test_get_predict_returns_browser_form(self) -> None:
        response = self.client.get("/predict")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Predict sentiment", response.text)
        self.assertIn('fetch("/predict"', response.text)

    def test_health_reports_ok(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_predict_uses_analyzer_output(self) -> None:
        response = self.client.post("/predict", json={"text": "works now"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["text"], "works now")
        self.assertEqual(payload["label"], "positive")


if __name__ == "__main__":
    unittest.main()

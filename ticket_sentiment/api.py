from __future__ import annotations

from collections.abc import Callable
import logging
from typing import TYPE_CHECKING

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "FastAPI support requires 'fastapi' and 'pydantic' to be installed."
    ) from exc

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .service import HybridTicketSentimentAnalyzer


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Ticket description")


def _default_analyzer_factory() -> HybridTicketSentimentAnalyzer:
    from .service import HybridTicketSentimentAnalyzer

    return HybridTicketSentimentAnalyzer()


def _browser_page() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ticket Sentiment Analyzer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #efe7da;
      --panel: #fffaf2;
      --ink: #1f1b16;
      --muted: #655d52;
      --border: #d2c6b5;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --error: #9a3412;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 34%),
        linear-gradient(180deg, #f6f1e8 0%, var(--bg) 100%);
      display: grid;
      place-items: center;
      padding: 24px;
    }

    main {
      width: min(760px, 100%);
      background: rgba(255, 250, 242, 0.95);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: 0 24px 70px rgba(31, 27, 22, 0.12);
      overflow: hidden;
    }

    .hero {
      padding: 28px 28px 12px;
      background: linear-gradient(135deg, rgba(15, 118, 110, 0.14), rgba(255, 255, 255, 0));
    }

    h1 {
      margin: 0 0 10px;
      font-size: clamp(2rem, 4vw, 3rem);
      line-height: 1;
    }

    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }

    form {
      padding: 24px 28px 28px;
      display: grid;
      gap: 14px;
    }

    label {
      font-weight: 600;
      font-size: 0.95rem;
    }

    textarea {
      width: 100%;
      min-height: 180px;
      padding: 16px;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: #fffdf9;
      color: var(--ink);
      font: inherit;
      resize: vertical;
    }

    textarea:focus {
      outline: 2px solid rgba(15, 118, 110, 0.18);
      border-color: var(--accent);
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }

    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      font-weight: 700;
      color: #f5fffd;
      background: linear-gradient(135deg, var(--accent), var(--accent-strong));
      cursor: pointer;
    }

    button[disabled] {
      opacity: 0.72;
      cursor: wait;
    }

    .link-row {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 0.94rem;
    }

    a {
      color: var(--accent-strong);
      text-decoration: none;
      font-weight: 600;
    }

    a:hover {
      text-decoration: underline;
    }

    .result,
    .error {
      margin: 0 28px 28px;
      border-radius: 18px;
      padding: 18px;
    }

    .result {
      background: rgba(15, 118, 110, 0.08);
      border: 1px solid rgba(15, 118, 110, 0.14);
    }

    .error {
      background: rgba(154, 52, 18, 0.08);
      border: 1px solid rgba(154, 52, 18, 0.16);
      color: var(--error);
    }

    .eyebrow {
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.78rem;
      font-weight: 700;
      color: var(--accent-strong);
      margin-bottom: 6px;
    }

    .result-header {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 10px;
    }

    .result-header h2 {
      margin: 0;
      font-size: 1.6rem;
      text-transform: capitalize;
    }

    .result-header span {
      font-weight: 700;
      color: var(--muted);
    }

    details {
      margin-top: 14px;
    }

    pre {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 12px 0 0;
      padding: 14px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.7);
      border: 1px solid rgba(15, 118, 110, 0.1);
      font-size: 0.92rem;
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Ticket Sentiment Analyzer</h1>
      <p>Paste a support ticket description below to predict whether the tone is negative, neutral, or positive.</p>
    </section>
    <form id="predict-form">
      <label for="ticket-text">Ticket description</label>
      <textarea id="ticket-text" name="text" placeholder="Example: Users still cannot access the dashboard after the latest deployment." required>Users still cannot access the dashboard after the latest deployment.</textarea>
      <div class="actions">
        <button id="submit-button" type="submit">Predict sentiment</button>
        <div class="link-row">
          <a href="/docs">Open API docs</a>
          <a href="/health" target="_blank" rel="noreferrer">Health check</a>
        </div>
      </div>
    </form>
    <section id="result" class="result" hidden>
      <div class="eyebrow">Prediction</div>
      <div class="result-header">
        <h2 id="result-label"></h2>
        <span id="result-confidence"></span>
      </div>
      <p id="result-rationale"></p>
      <details>
        <summary>Raw response</summary>
        <pre id="result-json"></pre>
      </details>
    </section>
    <p id="error" class="error" hidden></p>
  </main>
  <script>
    const form = document.getElementById("predict-form");
    const textArea = document.getElementById("ticket-text");
    const submitButton = document.getElementById("submit-button");
    const resultBox = document.getElementById("result");
    const resultLabel = document.getElementById("result-label");
    const resultConfidence = document.getElementById("result-confidence");
    const resultRationale = document.getElementById("result-rationale");
    const resultJson = document.getElementById("result-json");
    const errorBox = document.getElementById("error");

    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      const text = textArea.value.trim();
      if (!text) {
        errorBox.hidden = false;
        errorBox.textContent = "Enter a ticket description before predicting.";
        resultBox.hidden = true;
        return;
      }

      submitButton.disabled = true;
      submitButton.textContent = "Predicting...";
      errorBox.hidden = true;

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ text })
        });

        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Prediction failed.");
        }

        resultLabel.textContent = payload.label;
        resultConfidence.textContent = "Confidence: " + Number(payload.confidence).toFixed(3);
        resultRationale.textContent = payload.rationale || "No rationale returned.";
        resultJson.textContent = JSON.stringify(payload, null, 2);
        resultBox.hidden = false;
      } catch (error) {
        resultBox.hidden = true;
        errorBox.hidden = false;
        errorBox.textContent = error.message || "Prediction failed.";
      } finally {
        submitButton.disabled = false;
        submitButton.textContent = "Predict sentiment";
      }
    });
  </script>
</body>
</html>
"""


def create_app(
    analyzer_factory: Callable[[], HybridTicketSentimentAnalyzer] | None = None,
) -> FastAPI:
    analyzer_factory = analyzer_factory or _default_analyzer_factory
    analyzer: HybridTicketSentimentAnalyzer | None = None
    analyzer_error: Exception | None = None
    app = FastAPI(title="Ticket Sentiment Analyzer", version="0.1.0")

    def get_analyzer() -> HybridTicketSentimentAnalyzer:
        nonlocal analyzer, analyzer_error
        if analyzer is None:
            if analyzer_error is not None:
                raise RuntimeError("Ticket sentiment analyzer could not be initialized.") from analyzer_error
            try:
                analyzer = analyzer_factory()
            except Exception as exc:
                analyzer_error = exc
                logger.exception("Failed to initialize ticket sentiment analyzer")
                raise RuntimeError("Ticket sentiment analyzer could not be initialized.") from exc
        return analyzer

    @app.get("/", response_class=HTMLResponse)
    @app.get("/predict", response_class=HTMLResponse, include_in_schema=False)
    def root() -> HTMLResponse:
        return HTMLResponse(_browser_page())

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, object]:
        try:
            result = get_analyzer().predict(request.text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return result.as_dict()

    return app


app = create_app()

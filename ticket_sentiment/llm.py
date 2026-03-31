from __future__ import annotations

import json
import urllib.error
import urllib.request

from .scoring import clamp, label_to_score
from .types import RetrievedExample, Signal


SYSTEM_PROMPT = """You classify ticket sentiment into exactly one label: negative, neutral, or positive.
Use the user's emotional tone about the issue or support experience, not just technical severity.
Handle informal English, slang, spelling mistakes, abbreviations, emojis, and mixed wording.
Rules:
- negative: frustration, dissatisfaction, blockage, anger, disappointment, complaints, repeated failure
- positive: appreciation, praise, satisfaction, confirmed fix, smooth experience
- neutral: factual updates, clarification requests, status checks, mixed or weak emotion
Return strict JSON with keys: label, confidence, rationale.
confidence must be a number between 0 and 1.
"""


class OpenAIJudge:
    def __init__(self, api_key: str | None, model: str | None) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None
        self._load_attempted = False

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._client is not None and bool(self.model)

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if not self.api_key:
            return
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        except Exception:
            self._client = None

    def predict(
        self,
        text: str,
        normalized_text: str,
        signals: list[Signal],
        retrieved_examples: list[RetrievedExample],
    ) -> Signal | None:
        self._ensure_loaded()
        if self._client is None or not self.model:
            return None

        prompt = _build_user_prompt(text, normalized_text, signals, retrieved_examples)
        raw_response = None

        try:
            response = self._client.responses.create(
                model=self.model,
                temperature=0,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
                ],
            )
            raw_response = getattr(response, "output_text", None)
        except Exception:
            raw_response = None

        if not raw_response:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw_response = response.choices[0].message.content
            except Exception:
                return None

        try:
            payload = json.loads(raw_response)
            label = str(payload["label"]).strip().lower()
            if label not in {"negative", "neutral", "positive"}:
                return None
            confidence = clamp(float(payload.get("confidence", 0.7)))
            rationale = str(payload.get("rationale", "")).strip()
        except Exception:
            return None

        return Signal(
            source="llm",
            label=label,
            confidence=confidence,
            score=0.0 if label == "neutral" else label_to_score(label, confidence),
            details={"provider": "openai", "model": self.model, "rationale": rationale},
        )


class OllamaJudge:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    @property
    def available(self) -> bool:
        return bool(self.model)

    def predict(
        self,
        text: str,
        normalized_text: str,
        signals: list[Signal],
        retrieved_examples: list[RetrievedExample],
    ) -> Signal | None:
        if not self.model:
            return None

        prompt = f"{SYSTEM_PROMPT}\n\n{_build_user_prompt(text, normalized_text, signals, retrieved_examples)}"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        }

        request = urllib.request.Request(
            url=f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw_payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None

        raw_response = raw_payload.get("response")
        if not raw_response:
            return None

        try:
            parsed = json.loads(raw_response)
            label = str(parsed["label"]).strip().lower()
            if label not in {"negative", "neutral", "positive"}:
                return None
            confidence = clamp(float(parsed.get("confidence", 0.7)))
            rationale = str(parsed.get("rationale", "")).strip()
        except Exception:
            return None

        return Signal(
            source="llm",
            label=label,
            confidence=confidence,
            score=0.0 if label == "neutral" else label_to_score(label, confidence),
            details={"provider": "ollama", "model": self.model, "rationale": rationale},
        )


class CompositeLLMJudge:
    def __init__(
        self,
        provider: str,
        openai_api_key: str | None,
        openai_model: str | None,
        ollama_base_url: str,
        ollama_model: str,
    ) -> None:
        self.providers = []
        if provider in {"auto", "openai"}:
            self.providers.append(OpenAIJudge(openai_api_key, openai_model))
        if provider in {"auto", "ollama"}:
            self.providers.append(OllamaJudge(ollama_base_url, ollama_model))

    @property
    def available(self) -> bool:
        return any(provider.available for provider in self.providers)

    def predict(
        self,
        text: str,
        normalized_text: str,
        signals: list[Signal],
        retrieved_examples: list[RetrievedExample],
    ) -> Signal | None:
        for provider in self.providers:
            if not provider.available:
                continue
            signal = provider.predict(text, normalized_text, signals, retrieved_examples)
            if signal is not None:
                return signal
        return None


def _build_user_prompt(
    text: str,
    normalized_text: str,
    signals: list[Signal],
    retrieved_examples: list[RetrievedExample],
) -> str:
    signal_lines = "\n".join(
        f"- {signal.source}: label={signal.label}, confidence={signal.confidence:.3f}, score={signal.score:.3f}"
        for signal in signals
    )
    retrieved_lines = "\n".join(
        f"- similarity={example.similarity:.3f}, label={example.label}, text={example.text}"
        for example in retrieved_examples[:5]
    )

    return (
        "Ticket text:\n"
        f"{text}\n\n"
        "Normalized text:\n"
        f"{normalized_text}\n\n"
        "Current model signals:\n"
        f"{signal_lines or '- none'}\n\n"
        "Retrieved similar tickets:\n"
        f"{retrieved_lines or '- none'}\n\n"
        "Classify the sentiment now."
    )


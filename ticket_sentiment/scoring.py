from __future__ import annotations

from typing import Iterable

from .types import SentimentLabel, Signal


DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    "lexicon": 0.18,
    "rag": 0.26,
    "resolution_rule": 0.42,
    "transformer": 0.38,
    "vader": 0.12,
    "llm": 0.45,
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, value))


def score_to_label(score: float, neutral_window: float = 0.18) -> SentimentLabel:
    if score <= -neutral_window:
        return "negative"
    if score >= neutral_window:
        return "positive"
    return "neutral"


def label_to_score(label: SentimentLabel, confidence: float) -> float:
    if label == "negative":
        return -clamp(confidence)
    if label == "positive":
        return clamp(confidence)
    return 0.0


def _is_weak_neutral_signal(signal: Signal) -> bool:
    return signal.label == "neutral" and abs(signal.score) <= 0.05 and signal.confidence <= 0.3


def aggregate_signals(
    signals: Iterable[Signal],
    source_weights: dict[str, float] | None = None,
) -> dict[str, object]:
    weights = source_weights or DEFAULT_SOURCE_WEIGHTS
    contributions = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    strongest_signal_confidence = 0.0

    for signal in signals:
        base_weight = weights.get(signal.source, 0.16)
        signal_strength = clamp(signal.confidence, 0.1, 1.0)
        strongest_signal_confidence = max(strongest_signal_confidence, signal.confidence)
        polarity = clamp_signed(signal.score)
        weight = base_weight * signal_strength

        if _is_weak_neutral_signal(signal):
            weight *= 0.15

        polarity_strength = abs(polarity)
        resolved_strength = max(0.28, polarity_strength)

        if signal.label == "positive":
            contributions["positive"] += weight * (0.65 + (0.35 * resolved_strength))
        elif signal.label == "negative":
            contributions["negative"] += weight * (0.65 + (0.35 * resolved_strength))
        else:
            contributions["neutral"] += weight * (0.7 + (0.3 * max(0.0, 1.0 - polarity_strength)))

        residual_uncertainty = max(0.0, 0.45 - polarity_strength) / 0.45 if polarity_strength < 0.45 else 0.0
        if signal.label != "neutral":
            contributions["neutral"] += weight * 0.12 * residual_uncertainty

    total = sum(contributions.values()) or 1.0
    ordered = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
    label = ordered[0][0]
    top_score = ordered[0][1]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    dominance = top_score / total
    margin = (top_score - second_score) / total
    confidence = clamp(
        0.3 + (0.24 * dominance) + (0.18 * margin) + (0.18 * strongest_signal_confidence),
        0.0,
        0.96,
    )

    return {
        "label": label,
        "confidence": confidence,
        "margin": margin,
        "contributions": contributions,
    }

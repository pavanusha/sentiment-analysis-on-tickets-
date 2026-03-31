from __future__ import annotations

import math

from .preprocessing import extract_style_markers, tokenize
from .scoring import clamp, score_to_label
from .types import Signal


NEGATIONS = {"no", "not", "never", "without", "hardly", "cannot"}
INTENSIFIERS = {
    "very": 1.2,
    "extremely": 1.35,
    "super": 1.25,
    "really": 1.15,
    "so": 1.1,
    "too": 1.15,
}
DIMINISHERS = {
    "slightly": 0.7,
    "somewhat": 0.8,
    "maybe": 0.85,
    "perhaps": 0.85,
}

PHRASE_WEIGHTS = {
    "not working": -2.4,
    "still not working": -2.9,
    "does not work": -2.5,
    "cannot log in": -2.7,
    "cannot sign in": -2.7,
    "unable to log in": -2.7,
    "unable to access": -2.5,
    "not loading": -2.2,
    "not usable": -2.7,
    "server is down": -3.0,
    "service is down": -3.0,
    "down again": -2.4,
    "keeps failing": -2.6,
    "keeps crashing": -2.8,
    "very slow": -1.8,
    "super slow": -1.8,
    "data loss": -3.0,
    "critical issue": -2.9,
    "totally blocked": -3.0,
    "completely blocked": -3.0,
    "this is broken": -2.5,
    "frustrated with": -2.2,
    "team is mad": -2.4,
    "mad about it": -2.1,
    "not resolved": -2.6,
    "still broken": -2.8,
    "thanks resolved": 2.1,
    "issue fixed": 2.3,
    "now working": 2.1,
    "works now": 2.1,
    "working now": 2.1,
    "working fine": 2.0,
    "all good": 1.9,
    "looks good": 1.5,
    "much better": 1.8,
    "much better now": 2.1,
    "better now": 1.9,
    "sorted out": 2.0,
    "thank you": 1.4,
    "thanks team": 1.7,
    "appreciate the quick fix": 2.5,
    "resolved quickly": 2.3,
    "fixed quickly": 2.2,
    "no more errors": 2.0,
    "just checking": 0.0,
    "need clarification": 0.0,
    "need an update": -0.6,
    "please help": -0.7,
    "as soon as possible": -0.9,
}

TOKEN_WEIGHTS = {
    "blocked": -2.4,
    "broken": -2.1,
    "crash": -2.4,
    "crashing": -2.6,
    "down": -1.9,
    "failed": -2.1,
    "failing": -2.2,
    "failure": -2.1,
    "error": -1.8,
    "errors": -1.9,
    "urgent": -1.6,
    "critical": -1.9,
    "outage": -2.7,
    "slow": -1.4,
    "lag": -1.3,
    "lagging": -1.5,
    "stuck": -1.8,
    "useless": -2.6,
    "worst": -2.6,
    "frustrated": -2.3,
    "frustrating": -2.2,
    "mad": -2.2,
    "annoying": -1.7,
    "irritating": -2.0,
    "angry": -2.4,
    "terrible": -2.6,
    "disappointed": -2.0,
    "painful": -1.8,
    "ridiculous": -2.2,
    "unacceptable": -2.5,
    "worse": -1.9,
    "worrying": -1.7,
    "tired": -1.3,
    "better": 0.9,
    "good": 1.0,
    "great": 1.5,
    "awesome": 2.1,
    "perfect": 2.0,
    "resolved": 1.8,
    "fixed": 1.7,
    "stable": 1.3,
    "smooth": 1.5,
    "fast": 1.2,
    "thanks": 1.3,
    "appreciate": 1.6,
    "helpful": 1.4,
    "happy": 1.7,
    "love": 1.9,
    "nice": 1.1,
    "question": 0.0,
    "clarification": 0.0,
    "follow": 0.0,
    "update": 0.0,
}


def lexicon_signal(normalized_text: str, original_text: str | None = None) -> Signal:
    raw_score = 0.0
    matches: list[str] = []
    text = normalized_text

    for phrase, weight in PHRASE_WEIGHTS.items():
        if phrase in text:
            raw_score += weight
            matches.append(phrase)

    tokens = tokenize(text)
    for index, token in enumerate(tokens):
        if token not in TOKEN_WEIGHTS:
            continue

        token_score = TOKEN_WEIGHTS[token]
        left_context = tokens[max(0, index - 3) : index]

        if any(context_token in NEGATIONS for context_token in left_context):
            token_score *= -1.0

        for context_token in left_context[-2:]:
            if context_token in INTENSIFIERS:
                token_score *= INTENSIFIERS[context_token]
            if context_token in DIMINISHERS:
                token_score *= DIMINISHERS[context_token]

        raw_score += token_score
        matches.append(token)

    style_markers = extract_style_markers(original_text or normalized_text)
    if style_markers["has_negative_emoji"] or style_markers["has_angry_emoji"]:
        raw_score -= 1.0
    if style_markers["has_positive_emoji"]:
        raw_score += 0.8
    if style_markers["exclamations"] >= 2 and raw_score < 0:
        raw_score *= 1.1
    if style_markers["urgency_hits"] and raw_score <= 0:
        raw_score -= 0.4 * style_markers["urgency_hits"]
    if style_markers["uppercase_ratio"] >= 0.4 and raw_score < 0:
        raw_score *= 1.05

    bounded_score = math.tanh(raw_score / 6.0)
    label = score_to_label(bounded_score)

    confidence = clamp(
        0.22 + (0.5 * abs(bounded_score)) + (0.03 * min(len(matches), 8)),
        0.0,
        0.95,
    )
    if not matches:
        label = "neutral"
        bounded_score = 0.0
        confidence = 0.22

    return Signal(
        source="lexicon",
        label=label,
        confidence=confidence,
        score=bounded_score,
        details={"matches": matches[:10], "style_markers": style_markers},
    )

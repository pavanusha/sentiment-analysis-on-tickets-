from __future__ import annotations

import re
from collections import Counter


CONTRACTIONS = {
    "ain't": "is not",
    "aint": "is not",
    "can't": "cannot",
    "cant": "cannot",
    "won't": "will not",
    "wont": "will not",
    "don't": "do not",
    "dont": "do not",
    "didn't": "did not",
    "didnt": "did not",
    "doesn't": "does not",
    "doesnt": "does not",
    "isn't": "is not",
    "isnt": "is not",
    "aren't": "are not",
    "arent": "are not",
    "wasn't": "was not",
    "wasnt": "was not",
    "weren't": "were not",
    "werent": "were not",
    "couldn't": "could not",
    "couldnt": "could not",
    "shouldn't": "should not",
    "shouldnt": "should not",
    "wouldn't": "would not",
    "wouldnt": "would not",
    "hasn't": "has not",
    "hasnt": "has not",
    "haven't": "have not",
    "havent": "have not",
    "i'm": "i am",
    "im": "i am",
    "it's": "it is",
    "thx": "thanks",
    "thnx": "thanks",
    "thanx": "thanks",
    "ty": "thank you",
}

SLANG_MAP = {
    "pls": "please",
    "pls.": "please",
    "plz": "please",
    "asap": "as soon as possible",
    "btw": "by the way",
    "idk": "i do not know",
    "imo": "in my opinion",
    "fyi": "for your information",
    "u": "you",
    "ur": "your",
    "msg": "message",
    "appt": "application",
    "app": "application",
    "borked": "broken",
    "busted": "broken",
    "wtf": "very upset",
    "ugh": "frustrated",
    "meh": "unimpressed",
    "omg": "surprised",
    "laggy": "slow",
    "buggy": "unstable",
    "crashin": "crashing",
    "signin": "sign in",
    "login": "log in",
    "logout": "log out",
    "wanna": "want to",
    "gonna": "going to",
    "lemme": "let me",
    "kinda": "kind of",
    "sorta": "sort of",
    "tho": "though",
    "bc": "because",
    "cuz": "because",
    "coz": "because",
    "rn": "right now",
    "tmrw": "tomorrow",
    "yday": "yesterday",
    "ya": "yes",
    "yup": "yes",
    "nope": "no",
    "ppl": "people",
}

EMOJI_MAP = {
    "<3": " positive_emoji ",
    ">:(": " angry_emoji ",
    "\U0001F603": " positive_emoji ",
    "\U0001F600": " positive_emoji ",
    "\U0001F642": " positive_emoji ",
    "\U0001F60A": " positive_emoji ",
    "\U0001F44D": " positive_emoji ",
    "\U0001F64F": " appreciate_emoji ",
    "\U0001F61E": " negative_emoji ",
    "\U0001F621": " angry_emoji ",
    "\U0001F62D": " negative_emoji ",
    "\U0001F926": " frustrated_emoji ",
    "\U0001F624": " frustrated_emoji ",
    "\U0001F44E": " negative_emoji ",
    "\U0001F615": " frustrated_emoji ",
    ":)": " positive_emoji ",
    ":-)": " positive_emoji ",
    ":d": " positive_emoji ",
    ":-d": " positive_emoji ",
    ":(": " negative_emoji ",
    ":-(": " negative_emoji ",
    ":'(": " negative_emoji ",
    "😃": " positive_emoji ",
    "😀": " positive_emoji ",
    "🙂": " positive_emoji ",
    "😊": " positive_emoji ",
    "👍": " positive_emoji ",
    "🙏": " appreciate_emoji ",
    "😞": " negative_emoji ",
    "😡": " angry_emoji ",
    "😭": " negative_emoji ",
    "🤦": " frustrated_emoji ",
    "😤": " frustrated_emoji ",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.\-]+@[\w.\-]+\.\w+\b")
WORD_RE = re.compile(r"[a-z0-9_']+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
WHITESPACE_RE = re.compile(r"\s+")


def _replace_with_map(text: str, replacements: dict[str, str]) -> str:
    updated = text
    for source, target in replacements.items():
        updated = updated.replace(source, target)
    return updated


def _expand_word_map(text: str, mapping: dict[str, str]) -> str:
    updated = text
    for source, target in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        updated = re.sub(rf"\b{re.escape(source)}\b", target, updated)
    return updated


def reduce_elongation(token: str) -> str:
    return REPEATED_CHAR_RE.sub(r"\1\1", token)


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def normalize_text(text: str) -> str:
    normalized = text.strip()
    normalized = _replace_with_map(normalized, EMOJI_MAP)
    normalized = URL_RE.sub(" URL ", normalized)
    normalized = EMAIL_RE.sub(" EMAIL ", normalized)
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    normalized = normalized.lower()
    normalized = _expand_word_map(normalized, CONTRACTIONS)
    normalized = _expand_word_map(normalized, SLANG_MAP)
    tokens = [reduce_elongation(token) for token in tokenize(normalized)]
    return WHITESPACE_RE.sub(" ", " ".join(tokens)).strip()


def extract_style_markers(text: str) -> dict[str, float]:
    raw_tokens = re.findall(r"[A-Za-z]{2,}", text)
    uppercase_ratio = 0.0
    if raw_tokens:
        uppercase_ratio = sum(1 for token in raw_tokens if token.isupper()) / len(raw_tokens)

    normalized = normalize_text(text)
    token_counts = Counter(tokenize(normalized))
    urgency_hits = sum(token_counts.get(term, 0) for term in {"urgent", "critical", "blocked"})

    return {
        "exclamations": text.count("!"),
        "questions": text.count("?"),
        "uppercase_ratio": round(uppercase_ratio, 4),
        "urgency_hits": urgency_hits,
        "has_positive_emoji": 1.0 if "positive_emoji" in normalized else 0.0,
        "has_negative_emoji": 1.0 if "negative_emoji" in normalized else 0.0,
        "has_angry_emoji": 1.0 if "angry_emoji" in normalized else 0.0,
    }

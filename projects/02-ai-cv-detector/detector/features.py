"""Hand-crafted "stylometric" features: measurable habits of a writer.

Every feature is a rate (per 100 words) or a ratio, never a raw count, so that a
longer CV doesn't automatically look different from a shorter one.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .text import sentences, words

# Words LLMs reach for far more often than people do in CVs.
BUZZWORDS = frozenset(
    """spearheaded leveraged leveraging orchestrated fostered fostering streamlined
    championed pivotal dynamic innovative robust seamless seamlessly synergy
    passionate adept meticulous proactive dedicated versatile impactful
    comprehensive strategic strategically holistic cutting-edge showcasing
    demonstrating collaborated optimized enhancing ensuring""".split()
)

# Multi-word clichés, matched on the normalised (lower-case) text.
CLICHES = (
    "proven track record",
    "results-driven",
    "detail-oriented",
    "fast-paced environment",
    "cross-functional",
    "strong communication",
    "passionate about",
    "highly motivated",
    "team player",
    "excellent communication",
    "problem-solving skills",
    "commitment to",
    "a keen eye",
    "drive growth",
    "stakeholder",
)

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or that the this to was were will with".split()
)
FIRST_PERSON = frozenset({"i", "me", "my", "mine", "myself"})


def mattr(tokens: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio: vocabulary variety that doesn't depend on length."""
    if len(tokens) <= window:
        return len(set(tokens)) / max(len(tokens), 1)
    ratios = [len(set(tokens[i : i + window])) / window for i in range(0, len(tokens) - window + 1, 10)]
    return float(np.mean(ratios))


def stylometric_features(text: str) -> dict[str, float]:
    """Compute features from *normalised* text (see text.normalize)."""
    tokens = words(text)
    n = max(len(tokens), 1)
    per100 = 100 / n
    sentence_lengths = np.array([len(words(s)) for s in sentences(text)] or [0])
    mean_len = sentence_lengths.mean()

    return {
        "buzzword_rate": sum(t in BUZZWORDS for t in tokens) * per100,
        "cliche_rate": sum(text.count(c) for c in CLICHES) * per100,
        "avg_word_length": float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
        "vocab_variety": mattr(tokens),
        "sentence_length_mean": float(mean_len),
        # "Burstiness": people mix short and long sentences; LLMs are more uniform.
        "sentence_length_cv": float(sentence_lengths.std() / mean_len) if mean_len else 0.0,
        "digit_rate": len(re.findall(r"\d+", text)) * per100,
        "percent_rate": text.count("%") * per100,
        "first_person_rate": sum(t in FIRST_PERSON for t in tokens) * per100,
        "stopword_rate": sum(t in STOPWORDS for t in tokens) * per100,
        "comma_rate": text.count(",") * per100,
        "em_dash_rate": text.count("—") * per100,
        "semicolon_colon_rate": (text.count(";") + text.count(":")) * per100,
    }


FEATURE_NAMES = list(stylometric_features("placeholder text.").keys())


def feature_frame(texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame([stylometric_features(t) for t in texts], columns=FEATURE_NAMES)

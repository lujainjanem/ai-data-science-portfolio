"""Streamlit app: streamlit run app.py"""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from detector.features import BUZZWORDS, CLICHES, FEATURE_NAMES, feature_frame
from detector.models import explain
from detector.text import normalize, words

MODEL_PATH = Path(__file__).resolve().parent / "models" / "detector.joblib"

READABLE = {
    "buzzword_rate": "Uses many typical AI buzzwords",
    "cliche_rate": "Uses many stock phrases",
    "avg_word_length": "Long, formal words",
    "vocab_variety": "Vocabulary variety",
    "sentence_length_mean": "Sentence length",
    "sentence_length_cv": "Sentence-length variation",
    "digit_rate": "Amount of numbers",
    "percent_rate": "Many percentages",
    "first_person_rate": "Use of I / my",
    "stopword_rate": "Share of small function words",
    "comma_rate": "Comma use",
    "em_dash_rate": "Em dashes (—)",
    "semicolon_colon_rate": "Colons / semicolons",
    "lm_logprob_mean": "Very predictable wording (GPT-2)",
    "lm_logprob_std": "Uniform predictability (GPT-2)",
}

st.set_page_config(page_title="AI CV Detector", page_icon="🔎")
st.title("🔎 AI-written CV detector")
st.caption("Estimates how likely a CV was written or heavily rewritten by AI, and shows why.")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()


@st.cache_resource
def load():
    return joblib.load(MODEL_PATH)


bundle = load()
model, feature_cols = bundle["model"], bundle["feature_cols"]

upload = st.file_uploader("Upload a CV (.pdf or .txt)", type=["pdf", "txt"])
pasted = st.text_area("…or paste the text", height=200)

text = pasted
if upload is not None:
    if upload.name.lower().endswith(".pdf"):
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(upload.getvalue())).pages)
    else:
        text = upload.getvalue().decode("utf-8", errors="ignore")

if text.strip():
    clean = normalize(text, bundle["max_words"])
    if len(words(clean)) < 80:
        st.warning("This text is very short, so the estimate will be unreliable.")

    row = pd.concat([pd.DataFrame({"text": [clean]}), feature_frame([clean])], axis=1)
    if any(c not in FEATURE_NAMES for c in feature_cols):  # model was trained with --perplexity
        from detector.perplexity import PerplexityScorer

        scorer = st.cache_resource(PerplexityScorer)()
        for name, value in scorer.score(clean).items():
            row[name] = value

    proba = float(model.predict_proba(row)[:, 1][0])
    st.metric("Estimated probability the CV is AI-written", f"{proba:.0%}")
    st.progress(proba)

    st.subheader("What pushed the score up")
    reasons = explain(model, row)
    if not reasons:
        st.write("Nothing stood out as AI-like.")
    for name, _ in reasons:
        st.write(f"• {READABLE.get(name, f'Uses the phrase “{name}”')}")

    found = sorted({w for w in words(clean) if w in BUZZWORDS} | {c for c in CLICHES if c in clean})
    if found:
        st.caption("Buzzwords and stock phrases found: " + ", ".join(found))

    fpr = bundle["test_metrics"]["false_positive_rate"]
    st.info(
        f"⚠️ This is a statistical estimate, not proof. On the test set, {fpr:.0%} of genuine human CVs "
        "were wrongly flagged. Detectors like this are known to be less reliable on short texts and on "
        "writing by non-native English speakers. Never reject a candidate based on this score alone."
    )

"""The candidate models, the metrics, and per-CV explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)


def build_models(feature_cols: list[str]) -> dict[str, Pipeline]:
    """Four models, from most interpretable to most flexible.

    Each pipeline takes a DataFrame with a `text` column plus the feature columns.
    """
    style = ColumnTransformer([("style", StandardScaler(), feature_cols)])
    words = ColumnTransformer([("words", tfidf(), "text")])
    both = ColumnTransformer([("style", StandardScaler(), feature_cols), ("words", tfidf(), "text")])
    return {
        "style_logreg": Pipeline([("features", style), ("clf", LogisticRegression(max_iter=2000))]),
        "style_boosting": Pipeline(
            [("features", ColumnTransformer([("style", "passthrough", feature_cols)])),
             ("clf", HistGradientBoostingClassifier(max_depth=3, random_state=0))]
        ),
        "words_logreg": Pipeline([("features", words), ("clf", LogisticRegression(max_iter=2000, C=4.0))]),
        "combined_logreg": Pipeline([("features", both), ("clf", LogisticRegression(max_iter=2000, C=1.0))]),
    }


def metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    humans = y_true == 0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),  # share of AI CVs caught
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if len(set(y_true)) == 2 else float("nan"),
        # The costly mistake: a real person's CV flagged as AI.
        "false_positive_rate": float(y_pred[humans].mean()) if humans.any() else float("nan"),
    }


def feature_names(model: Pipeline) -> np.ndarray:
    return np.array([name.split("__", 1)[1] for name in model.named_steps["features"].get_feature_names_out()])


def top_terms(model: Pipeline, n: int = 15) -> dict[str, list[str]]:
    """Strongest signals for each class in a linear model: the "shortcut audit".

    If these look like formatting leftovers (e.g. section headings) rather than
    writing style, the model is cheating and `text.normalize` needs another rule.
    """
    coefs = model.named_steps["clf"].coef_[0]
    names = feature_names(model)
    order = np.argsort(coefs)
    return {"ai": list(names[order[::-1][:n]]), "human": list(names[order[:n]])}


def explain(model: Pipeline, row: pd.DataFrame, n: int = 6) -> list[tuple[str, float]]:
    """Which features pushed this CV towards "AI"? (coefficient × feature value)."""
    values = model.named_steps["features"].transform(row)
    values = values.toarray()[0] if hasattr(values, "toarray") else np.asarray(values)[0]
    contributions = model.named_steps["clf"].coef_[0] * values
    names = feature_names(model)
    order = np.argsort(contributions)[::-1]
    return [(names[i], float(contributions[i])) for i in order[:n] if contributions[i] > 0]

"""Load human CVs and AI-generated CVs into one labelled table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .text import normalize

TEXT_COLUMNS = ("Resume_str", "resume_str", "Resume", "resume_text", "text")
ID_COLUMNS = ("ID", "id")
CATEGORY_COLUMNS = ("Category", "category")


def _pick(df: pd.DataFrame, options: tuple[str, ...], required: bool = True) -> str | None:
    for col in options:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"None of the columns {options} found. Columns are: {list(df.columns)}")
    return None


def load_human_csv(path: str | Path) -> pd.DataFrame:
    """Return columns: id, text, category. Works with the Kaggle 'Resume Dataset' layout."""
    raw = pd.read_csv(path)
    text_col = _pick(raw, TEXT_COLUMNS)
    id_col = _pick(raw, ID_COLUMNS, required=False)
    cat_col = _pick(raw, CATEGORY_COLUMNS, required=False)
    df = pd.DataFrame(
        {
            "id": raw[id_col].astype(str) if id_col else raw.index.astype(str),
            "text": raw[text_col].astype(str),
            "category": raw[cat_col].astype(str) if cat_col else "general",
        }
    )
    return df[df["text"].str.split().str.len() >= 80].reset_index(drop=True)


def load_generated(folder: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(folder).glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    if not rows:
        raise FileNotFoundError(f"No generated CVs in {folder}. Run generate.py first.")
    return pd.DataFrame(rows)


def build_dataset(
    human_csv: str | Path,
    generated_dir: str | Path,
    n_human: int | None = None,
    max_words: int = 350,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per CV with: text (normalised), label (1 = AI), group, generator, mode.

    `group` ties an AI "polished" CV to the human CV it was rewritten from, so the two
    always land on the same side of the train/test split (otherwise the test set would
    contain near-copies of training rows: data leakage).
    """
    ai = load_generated(generated_dir)
    humans = load_human_csv(human_csv)
    n_human = min(n_human or len(ai), len(humans))
    humans = humans.sample(n=n_human, random_state=seed)

    human_rows = pd.DataFrame(
        {
            "text": humans["text"],
            "label": 0,
            "group": "h" + humans["id"],
            "generator": "human",
            "mode": "human",
            "category": humans["category"],
        }
    )
    ai_rows = pd.DataFrame(
        {
            "text": ai["text"],
            "label": 1,
            "group": [f"h{src}" if isinstance(src, str) and src else f"ai{i}" for i, src in zip(ai["id"], ai["source_id"])],
            "generator": ai["generator"],
            "mode": ai["mode"],
            "category": ai["category"],
        }
    )
    data = pd.concat([human_rows, ai_rows], ignore_index=True)
    data["text"] = [normalize(t, max_words) for t in data["text"]]
    return data[data["text"].str.len() > 0].reset_index(drop=True)

"""Train and evaluate the AI-CV detectors.

    python train.py                 # style + word features
    python train.py --perplexity    # also GPT-2 predictability features (slower, needs transformers)

Prints three experiments:
  1. Model comparison on a held-out test set
  2. Scratch vs polished: which kind of AI CV is harder to catch?
  3. Unseen generator: train without one LLM, test on it (only with 2+ generators)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from detector.data import build_dataset
from detector.features import FEATURE_NAMES, feature_frame
from detector.models import build_models, metrics, top_terms

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MODEL_PATH = ROOT / "models" / "detector.joblib"
DEPLOYED_MODEL = "combined_logreg"  # linear, so every prediction can be explained


def add_features(data: pd.DataFrame, use_perplexity: bool) -> tuple[pd.DataFrame, list[str]]:
    data = pd.concat([data, feature_frame(list(data["text"]))], axis=1)
    cols = list(FEATURE_NAMES)
    if use_perplexity:
        from detector.perplexity import PERPLEXITY_FEATURES, PerplexityScorer

        scorer = PerplexityScorer(cache_file=ROOT / "data" / "cache" / "perplexity.json")
        lm = pd.DataFrame([scorer.score(t) for t in data["text"]], columns=PERPLEXITY_FEATURES)
        scorer.save_cache()
        data = pd.concat([data, lm], axis=1)
        cols += PERPLEXITY_FEATURES
    return data, cols


def split(data: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """~75/25 split that keeps each group (a human CV and its AI rewrite) together."""
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(data, data["label"], data["group"]))
    return data.iloc[train_idx], data.iloc[test_idx]


def table(rows: dict[str, dict], cols: list[str]) -> str:
    lines = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.2f}" if isinstance(m[c], float) else str(m[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human-csv", type=Path, default=ROOT / "data" / "human" / "Resume.csv")
    parser.add_argument("--generated", type=Path, default=ROOT / "data" / "generated")
    parser.add_argument("--n-human", type=int, default=None, help="default: as many as AI CVs (balanced)")
    parser.add_argument("--max-words", type=int, default=350)
    parser.add_argument("--perplexity", action="store_true")
    args = parser.parse_args()

    data = build_dataset(args.human_csv, args.generated, args.n_human, args.max_words)
    data, feature_cols = add_features(data, args.perplexity)
    train, test = split(data)
    print(f"{len(data)} CVs ({(data.label == 0).sum()} human, {(data.label == 1).sum()} AI) "
          f"→ {len(train)} train / {len(test)} test\n")

    # 1. Model comparison
    models = build_models(feature_cols)
    results, probas = {}, {}
    for name, model in models.items():
        model.fit(train, train["label"])
        probas[name] = model.predict_proba(test)[:, 1]
        results[name] = metrics(test["label"], probas[name])
    cols = ["accuracy", "f1", "roc_auc", "recall", "false_positive_rate"]
    print("## 1. Model comparison (test set)\n" + table(results, cols) + "\n")

    # 2. Which kind of AI CV is harder to catch? (recall = share caught)
    deployed = models[DEPLOYED_MODEL]
    test_pred = (probas[DEPLOYED_MODEL] >= 0.5).astype(int)
    by_mode = {
        mode: {"n": int(mask.sum()), "caught": float(test_pred[mask.values].mean())}
        for mode in sorted(test.loc[test.label == 1, "mode"].unique())
        for mask in [test["mode"] == mode]
    }
    print(f"## 2. Share of AI CVs caught, by type ({DEPLOYED_MODEL})\n" + table(by_mode, ["n", "caught"]) + "\n")

    # 3. Generalisation to an LLM the model never saw
    generators = sorted(data.loc[data.label == 1, "generator"].unique())
    unseen = {}
    if len(generators) > 1:
        for held_out in generators:
            tr = train[train["generator"] != held_out]
            te = test[(test["generator"] == held_out) | (test["label"] == 0)]
            model = build_models(feature_cols)[DEPLOYED_MODEL].fit(tr, tr["label"])
            unseen[held_out] = metrics(te["label"], model.predict_proba(te)[:, 1])
        print("## 3. Unseen generator (trained without it)\n" + table(unseen, ["recall", "false_positive_rate", "roc_auc"]) + "\n")
    else:
        print("## 3. Unseen generator: skipped (generate CVs with a second model to run this)\n")

    audit = top_terms(models["combined_logreg"])
    print("## Shortcut audit: strongest signals\n"
          f"AI:    {', '.join(audit['ai'])}\nHuman: {', '.join(audit['human'])}\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(
        {"models": results, "by_mode": by_mode, "unseen_generator": unseen, "top_terms": audit,
         "deployed_model": DEPLOYED_MODEL, "n_train": len(train), "n_test": len(test)}, indent=2))

    # Refit the deployed model on all data for the app.
    final = build_models(feature_cols)[DEPLOYED_MODEL].fit(data, data["label"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "feature_cols": feature_cols, "max_words": args.max_words,
                 "test_metrics": results[DEPLOYED_MODEL]}, MODEL_PATH)
    print(f"Saved results/metrics.json and {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

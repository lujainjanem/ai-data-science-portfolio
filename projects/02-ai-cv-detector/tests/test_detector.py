import json
import random

import pandas as pd
import pytest

import generate
import train
from detector.data import build_dataset
from detector.features import FEATURE_NAMES, mattr, stylometric_features
from detector.models import build_models, explain, metrics, top_terms
from detector.text import normalize

HUMAN_BITS = [
    "managed payroll for 40 staff", "fixed printers and network issues", "answered customer calls daily",
    "prepared monthly reports in excel", "trained two new hires", "handled cash and closing duties",
    "worked weekends during busy season", "ordered supplies for the office", "kept the warehouse organised",
]
AI_BITS = [
    "spearheaded cross-functional initiatives", "leveraged innovative solutions to drive growth",
    "results-driven professional with a proven track record", "fostered a dynamic and collaborative culture",
    "streamlined processes, enhancing efficiency by 25%", "passionate about delivering impactful outcomes",
    "meticulous and detail-oriented team player", "orchestrated seamless stakeholder engagement",
]


def fake_cv(bits, rng, n=30):
    return ". ".join(rng.choice(bits) for _ in range(n)) + "."


@pytest.fixture
def dataset_files(tmp_path):
    rng = random.Random(0)
    humans = pd.DataFrame(
        {"ID": range(60), "Resume_str": [f"SUMMARY  Company Name City , State " + fake_cv(HUMAN_BITS, rng) for _ in range(60)],
         "Category": rng.choices(["HR", "ACCOUNTANT"], k=60)}
    )
    csv = tmp_path / "Resume.csv"
    humans.to_csv(csv, index=False)
    gen = tmp_path / "generated"
    gen.mkdir()
    with (gen / "fake_llm.jsonl").open("w") as f:
        for i in range(60):
            mode = "polish" if i % 2 else "scratch"
            f.write(json.dumps({"id": f"x{i}", "text": "**Summary**\n- " + fake_cv(AI_BITS, rng), "generator": "fake-llm",
                                "mode": mode, "category": "HR", "source_id": str(i) if mode == "polish" else ""}) + "\n")
    return csv, gen


def test_normalize_removes_formatting_and_placeholders():
    raw = "## **Summary**\n- Spearheaded growth at Company Name, City, State [Your Name]"
    assert normalize(raw) == "summary spearheaded growth at ,"
    assert normalize("one two three four", max_words=2) == "one two"


def test_clean_generation_strips_chatty_wrappers():
    raw = "Here is a professional resume:\n\nSUMMARY\nGreat engineer.\n\nNote: customise this for each job."
    assert generate.clean_generation(raw) == "SUMMARY\nGreat engineer."


def test_features_are_rates_not_counts():
    short = stylometric_features("spearheaded projects. fixed bugs.")
    long = stylometric_features(" ".join(["spearheaded projects. fixed bugs."] * 20))
    assert set(short) == set(FEATURE_NAMES)
    assert short["buzzword_rate"] == pytest.approx(long["buzzword_rate"])


def test_mattr_is_one_for_all_unique_words():
    assert mattr([f"w{i}" for i in range(120)]) == 1.0


def test_build_dataset_is_balanced_and_groups_polished_cvs(dataset_files):
    csv, gen = dataset_files
    data = build_dataset(csv, gen)
    assert (data.label == 1).sum() == (data.label == 0).sum() == 60
    polished = data[(data.label == 1) & (data["mode"] == "polish")]
    assert polished["group"].str.startswith("h").all()
    assert not data["text"].str.contains(r"\*\*|company name").any()


def test_split_keeps_groups_together(dataset_files):
    csv, gen = dataset_files
    data, _ = train.add_features(build_dataset(csv, gen), use_perplexity=False)
    tr, te = train.split(data)
    assert not set(tr["group"]) & set(te["group"])


def test_models_train_and_explain(dataset_files):
    csv, gen = dataset_files
    data, cols = train.add_features(build_dataset(csv, gen), use_perplexity=False)
    tr, te = train.split(data)
    for name, model in build_models(cols).items():
        model.fit(tr, tr["label"])
        m = metrics(te["label"], model.predict_proba(te)[:, 1])
        assert m["accuracy"] > 0.9, name
    combined = build_models(cols)["combined_logreg"].fit(tr, tr["label"])
    ai_row = te[te.label == 1].head(1)
    assert explain(combined, ai_row)
    assert set(top_terms(combined)) == {"ai", "human"}


def test_metrics_false_positive_rate():
    m = metrics([0, 0, 1, 1], [0.9, 0.1, 0.8, 0.7])
    assert m["false_positive_rate"] == 0.5
    assert m["recall"] == 1.0

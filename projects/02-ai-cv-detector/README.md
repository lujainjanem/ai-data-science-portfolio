# 🔎 AI-Written CV Detector

A machine learning model that estimates whether a CV was written, or heavily rewritten, by AI, and **explains which patterns** triggered the score. It's built with a strong focus on the error that matters most: **wrongly flagging a real person's CV**.

> 🚧 **Status:** code complete and tested. Results will be added once the dataset has been generated.

## Why this is hard (and interesting)

- CVs are short and formulaic, even when humans write them.
- Many people use AI to *polish* a CV they wrote themselves. Is that "AI-written"? This project measures **from-scratch** and **polished** CVs separately.
- AI detectors are known to wrongly flag non-native English writers more often. A false accusation here could cost someone a job, so the app shows a probability and its reasons, never a verdict.

## How it works

```
Human CVs (Kaggle, pre-ChatGPT) ─┐
                                  ├─► normalise ─► features ─► classifier ─► P(AI) + reasons
AI CVs (generated with Ollama) ───┘                 │
                                   style: buzzwords, sentence-length variation, vocabulary variety …
                                   words: TF-IDF of words and word pairs
                                   (optional) predictability under GPT-2
```

1. **Data** (`generate.py`, `detector/data.py`). Human CVs come from a public dataset collected before ChatGPT existed, so they can't be AI-written. AI CVs are generated locally and for free with **Ollama**, in two modes:
   - **scratch:** "write a resume for a mid-level accountant"
   - **polish:** "make this real CV sound more professional"
2. **Normalisation** (`detector/text.py`). Strips markdown, bullets, capitalisation and placeholder text from **both** classes. Without this, the model could just learn "markdown = AI" (*shortcut learning*) and score ~100% while learning nothing about writing style.
3. **Features** (`detector/features.py`). 13 hand-crafted style measurements, each a rate per 100 words so CV length doesn't matter. Examples: buzzword rate ("spearheaded", "leveraged"), stock phrases ("proven track record"), sentence-length variation (people vary more than AI does), and use of "I"/"my".
4. **Models** (`detector/models.py`). Four models, from most to least interpretable: style + logistic regression, style + gradient boosting, words (TF-IDF) + logistic regression, and everything combined.
5. **Evaluation** (`train.py`):
   - **Model comparison:** accuracy, F1, ROC AUC, and the **false-positive rate** (humans wrongly flagged)
   - **Scratch vs polished:** which kind of AI CV is harder to catch?
   - **Unseen generator:** train without one LLM's CVs, then test on them. Does the detector generalise, or did it just memorise one model's habits?
   - **Shortcut audit:** prints the strongest signals, so you can check they're real style differences and not formatting leftovers
6. **Leakage control.** A polished CV and the human CV it came from are always kept on the **same side** of the train/test split (grouped splitting), so the model is never tested on a near-copy of something it trained on.

## Results

_To be added after running the pipeline._

## Quick start

```bash
cd projects/02-ai-cv-detector
pip install -r requirements.txt

# 1. Human CVs: download Resume.csv into data/human/ (see data/human/README.md)

# 2. AI CVs: free, local
ollama pull llama3.2
python generate.py --model llama3.2 --n 100     # 100 scratch + 100 polished
ollama pull qwen2.5:3b
python generate.py --model qwen2.5:3b --n 100   # a second generator, for the unseen-generator test

# 3. Train and evaluate
python train.py

# 4. Try it
streamlit run app.py

# Tests (no data or Ollama needed)
pytest
```

## Project structure

```
detector/
  text.py        # normalisation (removes formatting shortcuts)
  features.py    # 13 style features
  perplexity.py  # optional GPT-2 predictability features
  data.py        # loads and labels human + AI CVs, grouping for leakage control
  models.py      # the 4 models, metrics, explanations, shortcut audit
generate.py      # creates AI CVs with Ollama
train.py         # trains, evaluates, saves the model
app.py           # Streamlit app: upload a CV, get P(AI) + reasons
tests/           # unit tests on synthetic data
```

## Skills demonstrated

Dataset construction · feature engineering · classical ML (logistic regression, gradient boosting, TF-IDF) · evaluation design (false-positive rate, out-of-distribution testing, leakage control) · model explainability · responsible AI · local LLMs (Ollama)

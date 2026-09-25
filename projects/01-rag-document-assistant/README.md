# 📚 RAG Document Assistant

A question-answering assistant that answers **only from your documents**, and cites the exact passages it used. It includes an **evaluation harness** that measures retrieval quality, answer faithfulness and abstention.

> Built from scratch rather than with LangChain or LlamaIndex, so every step (chunking, BM25 scoring, embedding search, rank fusion, grounded generation, evaluation) is visible and testable.

## What it does

```
docs/*.md ──► chunker ──► ┌─ BM25 (sparse)  ─┐
                          │                  ├─► Reciprocal Rank Fusion ──► top-k chunks ──► Claude + Citations API ──► answer [1][2]
question ───────────────► └─ Embeddings (dense) ┘
```

1. **Chunking** (`rag/chunking.py`). Splits markdown on headings, then into overlapping word windows so each chunk keeps its section context.
2. **Retrieval** (`rag/retrievers.py`). Three interchangeable retrievers:
   - **BM25**, implemented from scratch in NumPy. Best at exact terms, names and rare keywords.
   - **Dense**, using cosine similarity over sentence embeddings (`all-MiniLM-L6-v2`), or an offline **LSA** (TF-IDF + SVD) embedder. Catches paraphrases.
   - **Hybrid**, which merges both ranked lists with **Reciprocal Rank Fusion**. RRF uses ranks only, so the two score scales never need calibrating.
3. **Grounded generation** (`rag/generator.py`). Each retrieved chunk is sent to Claude as a separate document with the **Citations API** enabled, so every claim in the answer maps back to a specific chunk and quoted text. The system prompt tells the model to abstain when the documents don't contain the answer.
4. **Evaluation** (`evaluate.py`). A hand-labelled set of 35 questions: 30 answerable and 5 that the corpus deliberately **can't** answer.

## Results

The results below are on the included sample corpus: a 7-document, ~2,800-word ML handbook split into 35 chunks. A question counts as a **hit** when a retrieved chunk contains the labelled evidence phrase.

| Retriever | hit@1 | hit@3 | hit@5 | MRR@5 |
|---|---|---|---|---|
| BM25 | 0.83 | 0.90 | 0.93 | 0.87 |
| Dense (LSA) | 0.87 | 0.93 | 0.93 | 0.90 |
| Hybrid (BM25 + LSA, RRF) | 0.83 | 0.93 | 0.93 | 0.88 |
| Dense (MiniLM) | 0.73 | **1.00** | **1.00** | 0.87 |
| **Hybrid (BM25 + MiniLM, RRF)** | **0.87** | **1.00** | **1.00** | **0.92** |

**Findings**
- **Hybrid with MiniLM is the best configuration** (the default in the app and CLI). It finds a relevant chunk in the top 3 for every question, and ranks it first 87% of the time.
- **Neural embeddings fix BM25's misses.** BM25 and LSA both miss the same two paraphrased questions, where the question and the source share almost no words:
  - *q04*: "What happens to the two error components when a model gets more complex?" The source says "lowers bias but raises variance".
  - *q27*: "Do I need to standardize inputs before training a decision tree?" The source says "Tree-based models … do not need scaling".

  MiniLM retrieves both.
- **But MiniLM alone is worse at ranking the best chunk first** (hit@1 0.73 vs 0.83 for BM25). It finds the right topic but is less precise about exact terms. Fusing the two with RRF keeps MiniLM's recall and BM25's precision, giving the best hit@1 and MRR of any setup.

_Caveat: 30 answerable questions over a small corpus. One question moves hit@k by 0.03, so small differences aren't significant. A larger corpus and question set is the main next step._

**End-to-end answer quality.** `python evaluate.py --generation` generates an answer for every question and has Claude grade each one, using structured outputs, on:
- **faithfulness**: whether every claim is supported by the retrieved context
- **answer accuracy**: agreement with the reference answer
- **abstention rate**: how often the system correctly says "not in the documents" on the 5 unanswerable questions
- **citation coverage**: how many answers include citations

Results are saved to `results/`.

## Quick start

```bash
cd projects/01-rag-document-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # only needed for generated answers

# Chat UI (upload your own .md/.txt files in the sidebar)
streamlit run app.py

# CLI
python ask.py "Why should a scaler only be fit on the training folds?"
python ask.py "What is LoRA?" --retriever bm25 --retrieve-only   # no API call

# Evaluation
python evaluate.py --embedder lsa   # retrieval only, fully offline
python evaluate.py                  # retrieval with MiniLM embeddings
python evaluate.py --generation     # + LLM-judged answers (calls the API)

# Tests (offline, no API key needed)
pytest
```

Without an API key, the app still runs and shows the retrieved passages.

## Project structure

```
rag/
  chunking.py     # document loading, section-aware overlapping chunks
  retrievers.py   # BM25, dense (MiniLM / LSA), hybrid RRF
  generator.py    # Claude call with per-chunk citations
  pipeline.py     # retrieve → generate
app.py            # Streamlit chat UI
ask.py            # CLI
evaluate.py       # retrieval metrics + LLM-as-judge
data/docs/        # sample corpus (swap in your own)
data/eval/        # labelled questions (questions.jsonl)
tests/            # unit tests (offline)
```

## Design decisions

- **Evidence-phrase labels instead of chunk IDs.** Each question is labelled with a short phrase that must appear in a relevant chunk, so the labels stay valid when chunk size or overlap changes. That makes chunking experiments cheap.
- **Unanswerable questions in the eval set.** A RAG system that always answers will hallucinate. Measuring abstention is what catches that.
- **Citations API over "please cite your sources" prompting.** Citations come back as structured data (document index plus quoted text), not free text the model might invent.
- **Retrieval evaluated separately from generation.** When an answer is wrong, the metrics show whether retrieval failed or the model did.

## Next steps

- [x] Compare BM25, LSA and MiniLM embeddings, alone and fused
- [ ] Chunk-size sweep (75 / 150 / 300 words)
- [ ] Add a cross-encoder re-ranker and measure the change in hit@1
- [ ] Try a larger real corpus (e.g., a university handbook or a set of papers) with 100+ questions
- [ ] Add PDF ingestion
- [ ] Deploy to Streamlit Community Cloud or Hugging Face Spaces and link the demo here

## Skills demonstrated

Information retrieval (BM25, dense embeddings, rank fusion) · LLM application design · prompt design for grounding and abstention · evaluation methodology (hit@k, MRR, LLM-as-judge) · Python packaging and testing · Streamlit

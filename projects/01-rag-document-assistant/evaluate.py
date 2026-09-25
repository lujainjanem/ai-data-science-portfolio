"""Evaluate the RAG system.

Retrieval (free, offline):
    python evaluate.py --embedder lsa
    python evaluate.py                      # neural embeddings (needs sentence-transformers)

End-to-end answers with an LLM judge (calls the Claude API, costs money):
    python evaluate.py --generation --retriever hybrid
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel

from rag.chunking import Chunk, build_chunks
from rag.generator import MODEL, generate_answer
from rag.pipeline import DEFAULT_DOCS
from rag.retrievers import RETRIEVER_KINDS, Retriever, build_retriever

ROOT = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "results"
KS = (1, 3, 5)


@dataclass
class Question:
    id: str
    question: str
    source: str | None  # None => unanswerable from the corpus
    evidence: str | None  # short phrase that must appear in a relevant chunk
    answer: str | None  # reference answer used by the judge


def load_questions(path: Path) -> list[Question]:
    with path.open(encoding="utf-8") as f:
        return [Question(**{k: row.get(k) for k in Question.__annotations__}) for row in map(json.loads, f) if row]


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(chunk: Chunk, question: Question) -> bool:
    return chunk.source == question.source and _squash(question.evidence) in _squash(chunk.text)


def first_relevant_rank(retriever: Retriever, question: Question, depth: int) -> int | None:
    for rank, (chunk, _) in enumerate(retriever.search(question.question, depth), start=1):
        if is_relevant(chunk, question):
            return rank
    return None


def retrieval_metrics(retriever: Retriever, questions: Sequence[Question]) -> dict:
    answerable = [q for q in questions if q.source]
    ranks = {q.id: first_relevant_rank(retriever, q, max(KS)) for q in answerable}
    metrics = {f"hit@{k}": sum(r is not None and r <= k for r in ranks.values()) / len(ranks) for k in KS}
    metrics["mrr@5"] = sum(1 / r for r in ranks.values() if r) / len(ranks)
    metrics["misses@5"] = sorted(qid for qid, r in ranks.items() if r is None)
    return metrics


class Verdict(BaseModel):
    faithful: bool  # every claim is supported by the retrieved context
    correct: Literal["correct", "partial", "incorrect", "abstained"]
    reasoning: str


JUDGE_PROMPT = """You are grading a retrieval-augmented QA system.

<question>{question}</question>
<reference_answer>{reference}</reference_answer>
<retrieved_context>
{context}
</retrieved_context>
<system_answer>{answer}</system_answer>

Grade the system answer:
- faithful: true if every factual claim in the system answer is supported by the retrieved context (an abstention counts as faithful).
- correct: "correct" if it matches the reference answer, "partial" if it is incomplete but not wrong, "incorrect" if it contradicts the reference or answers when the reference says the information is not available, "abstained" if it says the documents don't contain the answer.
Keep reasoning to one or two sentences."""


def judge(client, question: Question, context: str, answer: str) -> Verdict:
    reference = question.answer or "The documents do not contain this information."
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question.question, reference=reference, context=context, answer=answer)}],
        output_format=Verdict,
    )
    return response.parsed_output


def generation_metrics(retriever: Retriever, questions: Sequence[Question], top_k: int) -> tuple[dict, list]:
    import anthropic

    client = anthropic.Anthropic()
    rows = []
    for q in questions:
        chunks = [chunk for chunk, _ in retriever.search(q.question, top_k)]
        answer = generate_answer(q.question, chunks, client=client)
        context = "\n\n".join(f"[{c.label}]\n{c.text}" for c in chunks)
        verdict = judge(client, q, context, answer.text)
        rows.append({"id": q.id, "question": q.question, "answer": answer.text,
                     "citations": len(answer.citations), **verdict.model_dump()})
        print(f"  {q.id}: {verdict.correct:<9} faithful={verdict.faithful}")

    answerable = [r for r, q in zip(rows, questions) if q.source]
    unanswerable = [r for r, q in zip(rows, questions) if not q.source]
    metrics = {
        "faithfulness": sum(r["faithful"] for r in rows) / len(rows),
        "answer_accuracy": sum(r["correct"] == "correct" for r in answerable) / max(len(answerable), 1),
        "answer_accuracy_incl_partial": sum(r["correct"] in ("correct", "partial") for r in answerable)
        / max(len(answerable), 1),
        "abstention_rate_on_unanswerable": sum(r["correct"] == "abstained" for r in unanswerable)
        / max(len(unanswerable), 1),
        "cited_answers": sum(r["citations"] > 0 for r in answerable) / max(len(answerable), 1),
    }
    return metrics, rows


def markdown_table(results: dict[str, dict]) -> str:
    cols = [f"hit@{k}" for k in KS] + ["mrr@5"]
    lines = ["| Retriever | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.2f}" for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--embedder", choices=["minilm", "lsa"], default="minilm")
    parser.add_argument("--retrievers", nargs="+", choices=RETRIEVER_KINDS, default=list(RETRIEVER_KINDS))
    parser.add_argument("--chunk-words", type=int, default=150)
    parser.add_argument("--chunk-overlap", type=int, default=30)
    parser.add_argument("--generation", action="store_true", help="also grade LLM answers (calls the API)")
    parser.add_argument("--retriever", choices=RETRIEVER_KINDS, default="hybrid", help="retriever for --generation")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    chunks = build_chunks(args.docs, args.chunk_words, args.chunk_overlap)
    n_answerable = sum(q.source is not None for q in questions)
    print(f"{len(chunks)} chunks, {len(questions)} questions ({n_answerable} answerable)\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    results = {}
    for kind in args.retrievers:
        name = kind if kind == "bm25" else f"{kind} ({args.embedder})"
        results[name] = retrieval_metrics(build_retriever(kind, chunks, args.embedder), questions)

    print(markdown_table(results))
    for name, m in results.items():
        if m["misses@5"]:
            print(f"\n{name} missed in top 5: {', '.join(m['misses@5'])}")
    out = RESULTS_DIR / f"retrieval_{args.embedder}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out.relative_to(ROOT)}")

    if args.generation:
        print(f"\nGrading end-to-end answers with retriever={args.retriever}, k={args.top_k} ...")
        retriever = build_retriever(args.retriever, chunks, args.embedder)
        metrics, rows = generation_metrics(retriever, questions, args.top_k)
        print(json.dumps(metrics, indent=2))
        out = RESULTS_DIR / f"generation_{args.retriever}_{args.embedder}.json"
        out.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2))
        print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

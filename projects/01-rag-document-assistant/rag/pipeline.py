"""Glue: documents -> chunks -> retriever -> Claude answer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, build_chunks
from .generator import Answer, generate_answer
from .retrievers import Hit, Retriever, build_retriever

DEFAULT_DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


@dataclass
class RAGResult:
    question: str
    hits: list[Hit]
    answer: Answer


class RAGPipeline:
    def __init__(
        self,
        docs_dir: str | Path = DEFAULT_DOCS,
        retriever: str = "hybrid",
        embedder: str = "minilm",
        top_k: int = 5,
        chunk_words: int = 150,
        chunk_overlap: int = 30,
    ):
        self.chunks: list[Chunk] = build_chunks(docs_dir, chunk_words, chunk_overlap)
        self.retriever: Retriever = build_retriever(retriever, self.chunks, embedder=embedder)
        self.top_k = top_k

    def retrieve(self, question: str) -> list[Hit]:
        return self.retriever.search(question, self.top_k)

    def ask(self, question: str) -> RAGResult:
        hits = self.retrieve(question)
        answer = generate_answer(question, [chunk for chunk, _ in hits])
        return RAGResult(question, hits, answer)

"""Command-line interface: python ask.py "What is data leakage?" """

from __future__ import annotations

import argparse

from rag.pipeline import DEFAULT_DOCS, RAGPipeline
from rag.retrievers import RETRIEVER_KINDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about your documents.")
    parser.add_argument("question")
    parser.add_argument("--docs", default=DEFAULT_DOCS, help="folder of .md / .txt files")
    parser.add_argument("--retriever", choices=RETRIEVER_KINDS, default="hybrid")
    parser.add_argument("--embedder", choices=["minilm", "lsa"], default="minilm")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--retrieve-only", action="store_true", help="skip the LLM call")
    args = parser.parse_args()

    pipeline = RAGPipeline(args.docs, args.retriever, args.embedder, args.top_k)

    if args.retrieve_only:
        for rank, (chunk, score) in enumerate(pipeline.retrieve(args.question), start=1):
            print(f"{rank}. [{score:.3f}] {chunk.label}\n   {chunk.text[:200]}...\n")
        return

    result = pipeline.ask(args.question)
    print(result.answer.text, "\n")
    for cite in result.answer.citations:
        print(f"[{cite.number}] {cite.chunk.label}: \"{cite.cited_text.strip()}\"")


if __name__ == "__main__":
    main()

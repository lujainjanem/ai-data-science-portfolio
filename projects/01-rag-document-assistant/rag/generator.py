"""Answer generation with Claude, grounded in retrieved chunks via the Citations API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import anthropic

from .chunking import Chunk

MODEL = "claude-opus-5"
# Server-side fallback: if the model declines a request, the API retries it on a
# recommended substitute model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You answer questions using only the documents provided in the user's message.

- Base every claim on the documents and cite them. Do not add facts from outside knowledge.
- If the documents do not contain the answer, say that you could not find it in the provided documents, and stop.
- Be concise: a short paragraph, or a few bullet points when listing items."""

NO_ANSWER = "I couldn't find anything relevant in the documents."


@dataclass
class Citation:
    number: int
    chunk: Chunk
    cited_text: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    stop_reason: str | None = None
    model: str | None = None


def build_content(question: str, chunks: Sequence[Chunk]) -> list[dict]:
    """One document block per chunk (so citations point at a chunk), then the question."""
    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": chunk.text},
            "title": chunk.label,
            "citations": {"enabled": True},
        }
        for chunk in chunks
    ]
    return [*documents, {"type": "text", "text": question}]


def generate_answer(
    question: str,
    chunks: Sequence[Chunk],
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
) -> Answer:
    if not chunks:
        return Answer(text=NO_ANSWER)

    client = client or anthropic.Anthropic()
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_content(question, chunks)}],
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )

    if response.stop_reason == "refusal":
        return Answer(
            text="The model declined to answer this question.",
            stop_reason=response.stop_reason,
            model=response.model,
        )

    return _render(response, chunks)


def _render(response, chunks: Sequence[Chunk]) -> Answer:
    """Join text blocks and turn each citation into a numbered [n] marker."""
    parts: list[str] = []
    citations: list[Citation] = []
    numbers: dict[tuple[int, str], int] = {}

    for block in response.content:
        if block.type != "text":
            continue  # thinking / fallback blocks are not part of the answer
        parts.append(block.text)
        markers = []
        for cite in block.citations or []:
            index = getattr(cite, "document_index", None)
            if index is None or not 0 <= index < len(chunks):
                continue
            key = (index, cite.cited_text)
            if key not in numbers:
                numbers[key] = len(citations) + 1
                citations.append(Citation(numbers[key], chunks[index], cite.cited_text))
            if f"[{numbers[key]}]" not in markers:
                markers.append(f"[{numbers[key]}]")
        parts.append("".join(markers))

    return Answer(
        text="".join(parts).strip(),
        citations=citations,
        stop_reason=response.stop_reason,
        model=response.model,
    )

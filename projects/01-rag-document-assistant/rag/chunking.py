"""Load documents from disk and split them into overlapping, section-aware chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt"}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str  # file name the chunk came from
    title: str  # document title (first H1, or the file stem)
    section: str  # nearest heading above the chunk
    text: str

    @property
    def searchable(self) -> str:
        """Text used for indexing: headings add useful context to short chunks."""
        return f"{self.title}. {self.section}.\n{self.text}"

    @property
    def label(self) -> str:
        return f"{self.source} › {self.section}"


def load_documents(folder: str | Path) -> list[tuple[str, str]]:
    """Return (file name, text) pairs for every supported file under `folder`."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Document folder not found: {folder}")
    docs = [
        (path.name, path.read_text(encoding="utf-8"))
        for path in sorted(folder.rglob("*"))
        if path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not docs:
        raise ValueError(f"No .md or .txt files found in {folder}")
    return docs


def split_sections(text: str, default_title: str) -> tuple[str, list[tuple[str, str]]]:
    """Split markdown into (heading, body) sections. Returns (doc title, sections)."""
    title = default_title
    sections: list[tuple[str, str]] = []
    heading, lines = title, []

    for line in text.splitlines():
        match = HEADING_RE.match(line.strip())
        if match:
            if lines:
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = match.group(2).strip(), []
            if match.group(1) == "#" and title == default_title:
                title = heading
        else:
            lines.append(line)
    if lines:
        sections.append((heading, "\n".join(lines).strip()))

    return title, [(h, body) for h, body in sections if body]


def window_words(words: list[str], max_words: int, overlap: int) -> list[list[str]]:
    """Slide a window of `max_words` over `words`, stepping by `max_words - overlap`."""
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    if len(words) <= max_words:
        return [words]
    step = max_words - overlap
    windows = []
    for start in range(0, len(words), step):
        windows.append(words[start : start + max_words])
        if start + max_words >= len(words):
            break
    return windows


def chunk_document(
    source: str, text: str, max_words: int = 150, overlap: int = 30
) -> list[Chunk]:
    title, sections = split_sections(text, default_title=Path(source).stem)
    chunks = []
    for section, body in sections:
        for window in window_words(body.split(), max_words, overlap):
            chunks.append(
                Chunk(
                    id=f"{source}#{len(chunks)}",
                    source=source,
                    title=title,
                    section=section,
                    text=" ".join(window),
                )
            )
    return chunks


def build_chunks(folder: str | Path, max_words: int = 150, overlap: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in load_documents(folder):
        chunks.extend(chunk_document(source, text, max_words, overlap))
    return chunks

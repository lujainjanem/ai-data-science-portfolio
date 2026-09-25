"""Three retrievers behind one interface: sparse (BM25), dense (embeddings), and hybrid (RRF)."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Protocol, Sequence

import numpy as np

from .chunking import Chunk

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it its of on or "
    "that the this to was what when where which who why will with you your".split()
)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

Hit = tuple[Chunk, float]
Encoder = Callable[[list[str]], np.ndarray]


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 5) -> list[Hit]: ...


class BM25Retriever:
    """Okapi BM25, implemented from scratch so the scoring is transparent."""

    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(tokenize(c.searchable)) for c in self.chunks]
        self.lengths = np.array([sum(tf.values()) for tf in self.term_freqs], dtype=float)
        self.avg_length = self.lengths.mean() if len(self.chunks) else 0.0

        doc_freq: Counter[str] = Counter()
        for tf in self.term_freqs:
            doc_freq.update(tf.keys())
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_length)
        for term in set(tokenize(query)):
            if term not in self.idf:
                continue
            tf = np.array([freqs.get(term, 0) for freqs in self.term_freqs], dtype=float)
            scores += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return scores

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return _top_k(self.chunks, self.scores(query), k)


def sentence_transformer_encoder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Encoder:
    # Imported lazily: torch is heavy and BM25 works without it.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(texts, normalize_embeddings=True)


def lsa_encoder(chunks: Sequence[Chunk], dims: int = 64) -> Encoder:
    """Latent Semantic Analysis: TF-IDF followed by truncated SVD.

    A classic, fully offline "semantic" embedding. It captures co-occurring terms
    (e.g. "overfitting" ~ "variance") but is far weaker than neural embeddings.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, sublinear_tf=True)
    tfidf = vectorizer.fit_transform([c.searchable for c in chunks])
    svd = TruncatedSVD(n_components=min(dims, tfidf.shape[1] - 1, len(chunks) - 1), random_state=0)
    svd.fit(tfidf)
    return lambda texts: svd.transform(vectorizer.transform(texts))


def make_encoder(embedder: str, chunks: Sequence[Chunk]) -> Encoder:
    if embedder == "minilm":
        return sentence_transformer_encoder()
    if embedder == "lsa":
        return lsa_encoder(chunks)
    raise ValueError(f"Unknown embedder '{embedder}'. Choose minilm or lsa.")


class DenseRetriever:
    """Cosine similarity over normalised embeddings. The encoder is injectable for tests."""

    name = "dense"

    def __init__(self, chunks: Sequence[Chunk], encoder: Encoder | None = None):
        self.chunks = list(chunks)
        self.encoder = encoder or sentence_transformer_encoder()
        self.matrix = _normalise(np.asarray(self.encoder([c.searchable for c in self.chunks])))

    def search(self, query: str, k: int = 5) -> list[Hit]:
        query_vec = _normalise(np.asarray(self.encoder([query])))[0]
        return _top_k(self.chunks, self.matrix @ query_vec, k)


class HybridRetriever:
    """Reciprocal Rank Fusion: score = sum(1 / (rrf_k + rank)) across retrievers.

    RRF only uses ranks, so it needs no score calibration between BM25 and cosine.
    """

    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60, candidates: int = 20):
        self.retrievers = list(retrievers)
        self.rrf_k, self.candidates = rrf_k, candidates

    def search(self, query: str, k: int = 5) -> list[Hit]:
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for retriever in self.retrievers:
            for rank, (chunk, _) in enumerate(retriever.search(query, self.candidates), start=1):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank)
                by_id[chunk.id] = chunk
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
        return [(by_id[chunk_id], score) for chunk_id, score in ranked]


RETRIEVER_KINDS = ("bm25", "dense", "hybrid")


def build_retriever(
    kind: str,
    chunks: Sequence[Chunk],
    embedder: str = "minilm",
    encoder: Encoder | None = None,
) -> Retriever:
    """Build a retriever. `encoder` overrides `embedder` (used by tests)."""
    if kind == "bm25":
        return BM25Retriever(chunks)
    if kind not in RETRIEVER_KINDS:
        raise ValueError(f"Unknown retriever '{kind}'. Choose one of {RETRIEVER_KINDS}.")
    dense = DenseRetriever(chunks, encoder or make_encoder(embedder, chunks))
    return dense if kind == "dense" else HybridRetriever([BM25Retriever(chunks), dense])


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def _top_k(chunks: list[Chunk], scores: np.ndarray, k: int) -> list[Hit]:
    order = np.argsort(-scores, kind="stable")[:k]
    return [(chunks[i], float(scores[i])) for i in order]

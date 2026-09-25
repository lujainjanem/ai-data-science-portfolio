"""A small, readable retrieval-augmented generation (RAG) system."""

from .chunking import Chunk, build_chunks
from .pipeline import RAGPipeline

__all__ = ["Chunk", "RAGPipeline", "build_chunks"]

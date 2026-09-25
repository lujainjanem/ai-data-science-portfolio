from types import SimpleNamespace

import numpy as np
import pytest

from evaluate import DEFAULT_QUESTIONS, Question, is_relevant, load_questions
from rag.chunking import Chunk, chunk_document, split_sections, window_words
from rag.generator import _render, build_content, generate_answer
from rag.retrievers import BM25Retriever, HybridRetriever, build_retriever, tokenize

DOC = """# Pets

## Cats
Cats are independent animals that purr when content.

## Dogs
Dogs are loyal animals that bark at strangers.
"""


def make_chunks() -> list[Chunk]:
    return chunk_document("pets.md", DOC)


def test_split_sections_uses_h1_as_title():
    title, sections = split_sections(DOC, default_title="pets")
    assert title == "Pets"
    assert [heading for heading, _ in sections] == ["Cats", "Dogs"]


def test_window_words_overlaps_and_covers_everything():
    words = [str(i) for i in range(25)]
    windows = window_words(words, max_words=10, overlap=3)
    assert windows[0][-3:] == windows[1][:3]
    assert windows[-1][-1] == "24"
    with pytest.raises(ValueError):
        window_words(words, max_words=5, overlap=5)


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the F1-score?") == ["f1", "score"]


def test_bm25_ranks_matching_section_first():
    hits = BM25Retriever(make_chunks()).search("which animal will bark?", k=2)
    assert hits[0][0].section == "Dogs"


def fake_encoder(texts):
    """Bag-of-two-words 'embedding' so dense retrieval is testable offline."""
    return np.array([[t.lower().count("purr"), t.lower().count("bark")] for t in texts], dtype=float)


def test_dense_and_hybrid_retrievers_use_injected_encoder():
    chunks = make_chunks()
    dense = build_retriever("dense", chunks, encoder=fake_encoder)
    assert dense.search("purr", k=1)[0][0].section == "Cats"
    hybrid = build_retriever("hybrid", chunks, encoder=fake_encoder)
    assert isinstance(hybrid, HybridRetriever)
    assert hybrid.search("bark", k=1)[0][0].section == "Dogs"


def test_lsa_embedder_builds_without_network():
    retriever = build_retriever("dense", make_chunks() * 3, embedder="lsa")
    assert len(retriever.search("loyal dogs", k=2)) == 2


def test_unknown_retriever_is_rejected():
    with pytest.raises(ValueError):
        build_retriever("magic", make_chunks())


def test_build_content_puts_documents_before_question():
    content = build_content("Q?", make_chunks())
    assert [block["type"] for block in content] == ["document", "document", "text"]
    assert content[0]["citations"] == {"enabled": True}


def test_render_numbers_citations_and_ignores_other_blocks():
    chunks = make_chunks()
    cite = SimpleNamespace(document_index=1, cited_text="Dogs are loyal")
    response = SimpleNamespace(
        stop_reason="end_turn",
        model="test-model",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="Dogs are loyal", citations=[cite, cite]),
            SimpleNamespace(type="text", text=".", citations=None),
        ],
    )
    answer = _render(response, chunks)
    assert answer.text == "Dogs are loyal[1]."
    assert len(answer.citations) == 1
    assert answer.citations[0].chunk.section == "Dogs"


def test_generate_answer_without_chunks_skips_api_call():
    assert "couldn't find" in generate_answer("anything", [], client=object()).text


def test_eval_set_is_well_formed():
    questions = load_questions(DEFAULT_QUESTIONS)
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        assert (q.source is None) == (q.evidence is None) == (q.answer is None), q.id


def test_is_relevant_matches_evidence_across_whitespace():
    chunk = make_chunks()[1]
    question = Question("x", "?", "pets.md", "loyal\n animals", "a")
    assert is_relevant(chunk, question)

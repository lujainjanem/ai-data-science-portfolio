"""Streamlit chat UI: streamlit run app.py"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import anthropic
import streamlit as st

from rag.pipeline import DEFAULT_DOCS, RAGPipeline
from rag.retrievers import RETRIEVER_KINDS

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Indexing documents...")
def load_pipeline(docs_dir: str, retriever: str, embedder: str, top_k: int) -> RAGPipeline:
    return RAGPipeline(docs_dir, retriever=retriever, embedder=embedder, top_k=top_k)


def save_uploads(files) -> str:
    folder = Path(tempfile.mkdtemp(prefix="rag-docs-"))
    for file in files:
        (folder / Path(file.name).name).write_bytes(file.getvalue())
    return str(folder)


with st.sidebar:
    st.header("Settings")
    uploads = st.file_uploader("Your documents (.md, .txt)", type=["md", "txt"], accept_multiple_files=True)
    retriever = st.selectbox("Retriever", RETRIEVER_KINDS, index=RETRIEVER_KINDS.index("hybrid"))
    embedder = st.selectbox(
        "Embedder",
        ["minilm", "lsa"],
        help="minilm = neural sentence embeddings; lsa = offline TF-IDF + SVD",
        disabled=retriever == "bm25",
    )
    top_k = st.slider("Chunks to retrieve (k)", 1, 10, 5)
    st.caption("Using the sample ML handbook" if not uploads else f"{len(uploads)} uploaded file(s)")

docs_dir = save_uploads(uploads) if uploads else str(DEFAULT_DOCS)
try:
    pipeline = load_pipeline(docs_dir, retriever, embedder, top_k)
except (ImportError, OSError) as err:
    st.error(f"Could not load the '{embedder}' embedder ({err}). Install sentence-transformers or pick 'lsa'.")
    st.stop()

st.title("📚 RAG Document Assistant")
st.caption(f"{len(pipeline.chunks)} chunks indexed · answers are grounded in your documents, with citations")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

if question := st.chat_input("Ask a question about the documents"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            st.info("No API key found, so only retrieval is shown. Set ANTHROPIC_API_KEY for generated answers.")
            hits = pipeline.retrieve(question)
            reply = "**Top retrieved passages:**\n\n" + "\n\n".join(
                f"**{chunk.label}** (score {score:.3f})\n> {chunk.text[:300]}..." for chunk, score in hits
            )
        else:
            try:
                with st.spinner("Retrieving and answering..."):
                    result = pipeline.ask(question)
                sources = "\n".join(
                    f"{c.number}. **{c.chunk.label}**: “{c.cited_text.strip()}”" for c in result.answer.citations
                )
                reply = result.answer.text + (f"\n\n**Sources**\n\n{sources}" if sources else "")
                with st.expander("Retrieved chunks"):
                    for chunk, score in result.hits:
                        st.markdown(f"**{chunk.label}** · score {score:.3f}\n\n{chunk.text}")
            except anthropic.APIError as err:
                reply = f"The Claude API returned an error: {err}"
        st.markdown(reply)
    st.session_state.history.append({"role": "assistant", "content": reply})

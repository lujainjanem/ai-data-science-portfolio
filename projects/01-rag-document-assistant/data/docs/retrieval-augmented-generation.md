# Retrieval-Augmented Generation

## Why retrieval helps language models

Large language models only know what was in their training data, which has a cutoff date and rarely includes private documents. They can also hallucinate, producing fluent but unsupported statements. Retrieval-augmented generation (RAG) addresses both problems by retrieving relevant passages from an external knowledge source at question time and giving them to the model as context, so that answers are grounded in specific sources that can be cited and checked.

## Chunking documents

Documents are split into chunks before indexing. Chunks that are too large dilute the relevant sentence with unrelated text and waste context; chunks that are too small lose the surrounding context needed to interpret them. A common approach is to split on document structure such as headings and paragraphs, then use a fixed-size window with some overlap so that sentences near a boundary appear in two chunks.

## Sparse, dense and hybrid retrieval

Sparse retrieval methods such as BM25 score documents by exact term matches, weighted by how rare each term is and normalized by document length. They are fast, need no training, and excel at names, codes and rare keywords, but they miss paraphrases. Dense retrieval encodes queries and chunks as embedding vectors and ranks by cosine similarity, which captures meaning even when the wording differs. Hybrid retrieval combines both. Reciprocal rank fusion is a simple way to merge the ranked lists: each document receives a score of one divided by a constant plus its rank in each list, and the scores are summed.

## Re-ranking

A two-stage design first retrieves a generous candidate set cheaply and then re-orders it with a more expensive model. Cross-encoder re-rankers read the query and a candidate passage together, which is more accurate than comparing separately computed embeddings but too slow to run over the whole corpus.

## Evaluating a RAG system

Retrieval and generation should be evaluated separately. Retrieval quality is measured with metrics such as hit rate at k, the share of questions for which a relevant chunk appears in the top k results, and mean reciprocal rank (MRR). Generation quality is often judged along two axes: faithfulness, whether every claim in the answer is supported by the retrieved context, and answer correctness against a reference. A good evaluation set also includes questions the corpus cannot answer, to check that the system abstains instead of inventing an answer.

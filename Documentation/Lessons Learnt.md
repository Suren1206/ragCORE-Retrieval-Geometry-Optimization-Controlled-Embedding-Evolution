1. Improving a RAG system requires understanding the interaction between representation, retrieval, generation, and gating—not optimizing components in isolation.

2. Retrieval failures are often caused by ambiguity in vector space, not absence of information.

3. Contrastive learning improves separation of similar chunks, but does not add new knowledge to the corpus.

4. Manual contrastive learning is highly subjective and not scalable, and must be applied carefully and incrementally.

5. Similarity ≠ answerability; high cosine similarity does not guarantee that a chunk contains a complete answer.

6. Threshold-based gating is a mechanism limitation, not a failure of retriever or LLM.

7. LLM often behaves correctly by refusing to answer when context is insufficient; not all failures are LLM errors.

8. Embedding model transition (Nomic → SentenceTransformer) changes the vector space; metrics must be interpreted directionally, not numerically.

9. Chunking affects geometry significantly; larger chunks increase similarity and reduce variance due to semantic averaging.

10. Re-ranking is effective only in ambiguous cases and cannot fix missing knowledge.

11. Optimization should be selective by corpus type; not all corpora require contrastive learning.

12. Controlled experimentation (same queries, one change at a time) is critical for valid conclusions.

13. Contrastive learning impacts the entire embedding space globally, even when trained on a few cases.

14. Final system quality depends on alignment across retrieval, generation, and gating, not any single improvement.

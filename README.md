# ragCORE — Retrieval Geometry Optimization & Controlled Embedding Evolution  
*A structured approach to improving retrieval through chunking, ranking, and contrastive geometry shaping while exposing the limits of similarity-based gating*

---

## Overview

ragCORE is a structured experimentation framework designed to analyze and improve retrieval behavior in Retrieval-Augmented Generation (RAG) systems.

The project focuses on understanding how representation (chunking), embedding geometry, ranking, and gating interact to influence final system performance. Instead of treating retrieval as a black box, ragCORE adopts a controlled, measurement-driven approach to systematically diagnose and optimize retrieval quality.

---

## Objective

The objective of ragCORE is to:

- Understand retrieval behavior through geometric analysis  
- Improve retrieval precision through controlled interventions  
- Separate retrieval, generation, and gating effects  
- Establish a repeatable methodology for retrieval optimization  

---

## Phase 1 — Baseline Setup

- Narrative corpus (335 words) on diabetes  
- Chunking: 50 words with overlap  
- Embedding: Nomic (Ollama)  
- LLM: llama3.2:3b  
- Threshold: 0.75 (Top-1 chunk passed to LLM)

**Results:**

| Metric | Value |
|------|------|
| TP | 11 |
| FP | 6 |
| TN | 13 |
| FN | 5 |

**Effective Success Rate:** **30%**

**Observation:**  
Low embedding variance and high inter-chunk similarity indicated a tightly clustered and ambiguous vector space.

---

## Phase 2 — Cross Corpus Comparison

Three corpus styles were evaluated:

- Narrative  
- Rule-based  
- Q&A  

**Key Observations:**

- Better geometric separation (higher rank gap) did not guarantee better retrieval  
- Rule-based corpus showed higher False Positives despite strong separation  
- Retrieval performance depends on both structure and answer completeness  

---

## Phase 3 — Retrieval Optimization

### (a) Re-ranking

- Cross-encoder used for Top-N re-evaluation  
- Effective only in ambiguous cases  
- No impact when answer was absent  

**Conclusion:**  
Re-ranking improves ordering, not knowledge availability.

---

### (b) Re-chunking

Chunk sizes evaluated: **50 → 75 → 100**

**Results (Chunk 75):**

| Metric | Before | After |
|------|--------|-------|
| TP | 11 | 15 |
| FP | 6 | 2 |

**Success Rate:** **30% → 42%**

**Observation:**

- Larger chunks improved context  
- Diminishing returns beyond optimal size  
- Re-ranking impact reduced after re-chunking due to lower ambiguity  

---

## Phase 4 — Contrastive Learning

Applied selectively to 5 cases:

- 4 False Negatives  
- 1 False Positive  

**Results:**

| Metric | Before | After |
|------|--------|-------|
| TP | 15 | 19 |
| FN | 6 | 2 |

**Success Rate:** **42% → 54%**

**Key Observations:**

- Contrastive learning improved separation and resolved ambiguity  
- False Negatives were effectively converted  
- False Positives remained unchanged  

---

## Geometry Insights

### Before Contrastive Learning (Chunk 75)

- Embedding Variance: 0.000253  
- Inter-chunk Similarity: 0.7668  
- Rank Gap: 0.061  

### After Contrastive Learning

- Embedding Variance: 0.001370 ↑  
- Inter-chunk Similarity: 0.3685 ↓  
- Rank Gap: impacted by threshold classification  

**Interpretation:**

- Improved spread and separation in embedding space  
- Metrics must be interpreted directionally due to embedding model transition  

---

## Key Insights

- Retrieval ambiguity is the primary cause of errors  
- Chunking significantly influences embedding geometry  
- Re-ranking is effective only under ambiguity  
- Contrastive learning improves separation but does not add knowledge  
- Similarity does not guarantee answerability  
- Threshold-based gating has inherent limitations  
- Retrieval, generation, and gating must be analyzed independently  

---

## Architecture Perspective

ragCORE demonstrates that RAG performance is governed by:

- Representation (chunking)  
- Embedding geometry  
- Ranking behavior  
- Gating logic  
- Generation behavior  

System quality emerges from alignment across these layers.

---

## Methodology Principles

- Controlled experimentation (one change at a time)  
- Fixed query set for evaluation  
- Separation of system layers  
- Directional interpretation of metrics  
- Selective optimization based on observed behavior  

---

## Lessons Learnt

- Retrieval improvement is driven by reducing ambiguity, not increasing data  
- Contrastive learning is powerful but requires careful supervision  
- Similarity ≠ answerability  
- Gating is a mechanism limitation, not a model failure  
- Embedding model choice affects interpretation of results  
- Optimization must be selective and corpus-dependent  

---

## Conclusion

ragCORE establishes a structured approach to understanding and improving retrieval systems by focusing on embedding geometry and controlled interventions.

It highlights that true system improvement comes from aligning representation, retrieval, and generation rather than optimizing any single component in isolation.

---

## One-line Summary

👉 Retrieval improved from **30% → 42% → 54%** through chunking and contrastive learning, while exposing the limitations of similarity-based gating.

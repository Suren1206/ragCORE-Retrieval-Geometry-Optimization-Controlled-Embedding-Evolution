import json
import numpy as np
import faiss
from pathlib import Path


# ----------------------------------------------------------------------------------------------------
# Paths - Need to change the Index Path and Mapping Path accordingly for getting evaluation results for them
# 
# ----------------------------------------------------------------------------------------------------
BASE_PATH = Path("/home/surendran/LangChain/ragCORE")

INDEX_PATH = BASE_PATH / "vector_store/Baseline/Corpus3_FAQ/faiss_index.bin"
MAPPING_PATH = BASE_PATH / "vector_store/Baseline/Corpus3_FAQ/id_to_chunk.json"
QUERY_FILE = BASE_PATH / "Data/query_set_v1.txt"

OUTPUT_FILE = BASE_PATH / "Output/retrieval_results_corpus3.txt"


# -----------------------------
# Load model
# -----------------------------
def load_model():
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model="nomic-embed-text")


# -----------------------------
# Load queries
# -----------------------------
def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# -----------------------------
# Band classification
# -----------------------------
def get_band(score):
    if score >= 0.75:
        return "HIGH"
    elif score >= 0.60:
        return "MID"
    else:
        return "LOW"


# -----------------------------
# Detect FAISS metric safely
# -----------------------------
def detect_metric(index):
    metric_type = getattr(index, "metric_type", None)

    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return "IP"
    elif metric_type == faiss.METRIC_L2:
        return "L2"
    else:
        return "UNKNOWN"


# -----------------------------
# Convert FAISS output to score
# -----------------------------
def convert_to_score(raw_value, metric_kind):
    if metric_kind == "IP":
        # For normalized embeddings with inner product index,
        # raw_value is already cosine similarity-like score
        return float(raw_value)

    elif metric_kind == "L2":
        # For normalized embeddings with L2 distance:
        # cosine ~= 1 - (squared_l2 / 2)
        return float(1 - (raw_value / 2))

    else:
        # Fallback: return raw value as-is
        return float(raw_value)


# -----------------------------
# Main
# -----------------------------
def main():
    index = faiss.read_index(str(INDEX_PATH))

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        id_to_chunk = json.load(f)

    model = load_model()
    queries = load_queries(QUERY_FILE)

    metric_kind = detect_metric(index)

    print(f"Loaded FAISS index from: {INDEX_PATH}")
    print(f"FAISS index dimension: {index.d}")
    print(f"FAISS metric type: {metric_kind}")

    results_lines = []

    high_count = 0
    mid_count = 0
    low_count = 0
    gaps = []

    for q_idx, query in enumerate(queries):
        results_lines.append("\n==============================")
        results_lines.append(f"Query {q_idx + 1}: {query}")
        results_lines.append("------------------------------")

        # Encode query
        raw_q_emb = model.embed_query(query)
        q_emb = np.array([raw_q_emb], dtype="float32")

        # Dimension check
        if q_emb.shape[1] != index.d:
            raise ValueError(
                f"Embedding dimension mismatch for query {q_idx + 1}: "
                f"query dim = {q_emb.shape[1]}, index dim = {index.d}. "
                f"Use the same embedding model that was used to build the FAISS index."
            )

        # Normalize query embedding
        norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
        if np.any(norm == 0):
            raise ValueError(f"Zero vector encountered for query {q_idx + 1}: {query}")

        q_emb = q_emb / norm

        # Search
        distances, indices = index.search(q_emb, 3)

        top_scores = []

        for rank, (idx, raw_value) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:
                continue

            chunk = id_to_chunk.get(str(idx), "[Chunk not found in mapping]")
            score = convert_to_score(raw_value, metric_kind)
            band = get_band(score)

            top_scores.append(score)

            results_lines.append(
                f"Rank {rank + 1} | Raw: {raw_value:.4f} | Score: {score:.4f} | Band: {band}\n"
                f"Chunk: {chunk}\n"
            )

        # Top-1 summary count
        if top_scores:
            top1_score = top_scores[0]
            top1_band = get_band(top1_score)

            if top1_band == "HIGH":
                high_count += 1
            elif top1_band == "MID":
                mid_count += 1
            else:
                low_count += 1

        # Score gap
        if len(top_scores) >= 2:
            gap = top_scores[0] - top_scores[1]
            gaps.append(gap)
            results_lines.append(f"Score Gap (Top1 - Top2): {gap:.4f}")
        else:
            results_lines.append("Score Gap (Top1 - Top2): N/A")

    # Summary
    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    results_lines.append("\n==============================")
    results_lines.append("SUMMARY")
    results_lines.append("------------------------------")
    results_lines.append(f"Total Queries: {len(queries)}")
    results_lines.append(f"HIGH count: {high_count}")
    results_lines.append(f"MID count: {mid_count}")
    results_lines.append(f"LOW count: {low_count}")
    results_lines.append(f"Average Score Gap: {avg_gap:.4f}")
    results_lines.append(f"FAISS metric type used: {metric_kind}")
    results_lines.append(f"FAISS index dimension: {index.d}")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(results_lines))

    print("Retrieval evaluation completed. Results saved.")


if __name__ == "__main__":
    main()
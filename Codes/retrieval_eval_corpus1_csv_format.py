import json
import numpy as np
import faiss
import csv
from pathlib import Path


# -------------------------------------------------------------------
# Paths - same codes can be used for Rule based corpus and FAQ corpus
# -------------------------------------------------------------------
BASE_PATH = Path("/home/surendran/LangChain/ragCORE")

QUERY_FILE = BASE_PATH / "Data/query_set_v1.txt"
INDEX_PATH = BASE_PATH / "vector_store/Baseline/Corpus_Narrative_chunk50/faiss_index.bin"
MAPPING_PATH = BASE_PATH / "vector_store/Baseline/Corpus_Narrative_chunk50/id_to_chunk.json"

OUTPUT_FILE = BASE_PATH / "Output/retrieval_generation_corpus1.csv"


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
        return float(raw_value)
    elif metric_kind == "L2":
        return float(1 - (raw_value / 2))
    else:
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

    rows = []

    for q_idx, query in enumerate(queries):
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

        # Normalize
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

            rows.append({
                "query_id": q_idx + 1,
                "query": query,
                "rank": rank + 1,
                "raw_value": round(float(raw_value), 4),
                "score": round(score, 4),
                "band": band,
                "score_gap": "",
                "chunk": chunk
            })

        # Write score gap only on rank 1 row for this query
        if len(top_scores) >= 2:
            score_gap = round(top_scores[0] - top_scores[1], 4)

            for row in reversed(rows):
                if row["query_id"] == q_idx + 1 and row["rank"] == 1:
                    row["score_gap"] = score_gap
                    break

    # -----------------------------
    # Write CSV
    # -----------------------------
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "query",
                "rank",
                "raw_value",
                "score",
                "band",
                "score_gap",
                "chunk"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("CSV retrieval results saved.")


if __name__ == "__main__":
    main()
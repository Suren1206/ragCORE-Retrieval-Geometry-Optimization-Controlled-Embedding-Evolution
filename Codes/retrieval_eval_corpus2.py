import json
import numpy as np
import faiss
from pathlib import Path


# -----------------------------
# Paths (Corpus2)
# -----------------------------
BASE_PATH = Path("/home/surendran/LangChain/ragCORE")

INDEX_PATH = BASE_PATH / "vector_store/Baseline/Corpus2_Rule_based/faiss_index.bin"
MAPPING_PATH = BASE_PATH / "vector_store/Baseline/Corpus2_Rule_based/id_to_chunk.json"
QUERY_FILE = BASE_PATH / "Data/query_set_v1.txt"

OUTPUT_FILE = BASE_PATH / "Output/retrieval_results_corpus2.txt"



# -----------------------------
# Model
# -----------------------------
def load_model():
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model="nomic-embed-text")


# -----------------------------
# Helpers
# -----------------------------
def load_queries(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_band(score):
    if score >= 0.75:
        return "HIGH"
    elif score >= 0.60:
        return "MID"
    else:
        return "LOW"


# -----------------------------
# Main
# -----------------------------
def main():
    index = faiss.read_index(str(INDEX_PATH))

    with open(MAPPING_PATH, "r") as f:
        id_to_chunk = json.load(f)

    model = load_model()
    queries = load_queries(QUERY_FILE)

    results_lines = []

    high_count = 0
    gaps = []

    for q_idx, query in enumerate(queries):
        results_lines.append("\n==============================")
        results_lines.append(f"Query {q_idx+1}: {query}")
        results_lines.append("------------------------------")

        # Embed + normalize
        q_emb = model.embed_query(query)
        q_emb = np.array([q_emb]).astype("float32")
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        # Search Top-3
        distances, indices = index.search(q_emb, 3)

        top_scores = []

        # Process Top-3
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            chunk = id_to_chunk[str(idx)]

            score = 1 - (dist / 2)
            band = get_band(score)

            top_scores.append(score)

            results_lines.append(
                f"Rank {rank+1} | Score: {score:.4f} | Band: {band}\nChunk: {chunk}\n"
            )

        # -----------------------------
        # Top-1 HIGH count only
        # -----------------------------
        top1_score = top_scores[0]

        if top1_score >= 0.75:
            high_count += 1

        # Score gap
        gap = top_scores[0] - top_scores[1]
        gaps.append(gap)

        results_lines.append(f"Score Gap (Top1 - Top2): {gap:.4f}")

    # -----------------------------
    # Summary
    # -----------------------------
    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    results_lines.append("\n==============================")
    results_lines.append("SUMMARY (TOP-1 HIGH ONLY)")
    results_lines.append("------------------------------")
    results_lines.append(f"Total Queries: {len(queries)}")
    results_lines.append(f"HIGH (score ≥ 0.75): {high_count}")
    results_lines.append(f"Average Score Gap: {avg_gap:.4f}")

    # Save
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(results_lines))

    print("Corpus2 retrieval evaluation completed.")


if __name__ == "__main__":
    main()
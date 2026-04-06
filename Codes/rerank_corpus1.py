import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import CrossEncoder


# -----------------------------
# Paths - sane code can be used for corpus2 and corpus3 also
# -----------------------------
BASE_PATH = Path("/home/surendran/LangChain/ragCORE")

INDEX_PATH = BASE_PATH / "vector_store/Baseline/Corpus_Narrative_chunk50/faiss_index.bin"
MAPPING_PATH = BASE_PATH / "vector_store/Baseline/Corpus_Narrative_chunk50/id_to_chunk.json"
QUERY_FILE = BASE_PATH / "Data/query_set_v1.txt"

OUTPUT_FILE = BASE_PATH / "Output/rerank_corpus1_chunk50.txt"


# -----------------------------
# Retrieval model
# -----------------------------
def load_embedding_model():
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model="nomic-embed-text")


# -----------------------------
# Reranker model
# -----------------------------
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# -----------------------------
# Helpers
# -----------------------------
def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_band(score):
    if score >= 0.75:
        return "HIGH"
    elif score >= 0.60:
        return "MID"
    else:
        return "LOW"


def format_rank_block(rank, chunk, retrieval_score, band):
    return (
        f"Rank {rank} | Retrieval Score: {retrieval_score:.4f} | Band: {band}\n"
        f"Chunk: {chunk}\n"
    )


# -----------------------------
# Main
# -----------------------------
def main():
    index = faiss.read_index(str(INDEX_PATH))

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        id_to_chunk = json.load(f)

    embed_model = load_embedding_model()
    reranker = load_reranker()
    queries = load_queries(QUERY_FILE)

    lines = []

    for q_idx, query in enumerate(queries, start=1):
        lines.append("\n============================================================")
        lines.append(f"Query {q_idx}: {query}")
        lines.append("------------------------------------------------------------")

        # ---------------------------------
        # Step 1: Baseline FAISS retrieval
        # ---------------------------------
        q_emb = embed_model.embed_query(query)
        q_emb = np.array([q_emb]).astype("float32")
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        distances, indices = index.search(q_emb, 3)

        baseline_results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
            chunk = id_to_chunk[str(idx)]
            retrieval_score = 1 - (dist / 2)
            band = get_band(retrieval_score)

            baseline_results.append({
                "baseline_rank": rank,
                "chunk_id": int(idx),
                "chunk": chunk,
                "retrieval_score": float(retrieval_score),
                "band": band
            })

        baseline_top1_chunk = baseline_results[0]["chunk"]

        lines.append("BASELINE TOP 3")
        lines.append("--------------")
        for item in baseline_results:
            lines.append(
                format_rank_block(
                    item["baseline_rank"],
                    item["chunk"],
                    item["retrieval_score"],
                    item["band"]
                )
            )

        # ---------------------------------
        # Step 2: Cross-encoder reranking
        # ---------------------------------
        pairs = [(query, item["chunk"]) for item in baseline_results]
        rerank_scores = reranker.predict(pairs)

        reranked_results = []
        for item, rerank_score in zip(baseline_results, rerank_scores):
            enriched = item.copy()
            enriched["rerank_score"] = float(rerank_score)
            reranked_results.append(enriched)

        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)

        for new_rank, item in enumerate(reranked_results, start=1):
            item["reranked_rank"] = new_rank
            item["rank_shift"] = item["baseline_rank"] - new_rank

        reranked_top1_chunk = reranked_results[0]["chunk"]
        top1_changed = baseline_top1_chunk != reranked_top1_chunk

        lines.append("RERANKED TOP 3")
        lines.append("--------------")
        for item in reranked_results:
            lines.append(
                f"New Rank {item['reranked_rank']} | "
                f"Old Rank {item['baseline_rank']} | "
                f"Rerank Score: {item['rerank_score']:.4f} | "
                f"Rank Shift: {item['rank_shift']:+d}\n"
                f"Chunk: {item['chunk']}\n"
            )

        lines.append("SUMMARY FOR THIS QUERY")
        lines.append("----------------------")
        lines.append(f"Top-1 Changed: {'YES' if top1_changed else 'NO'}")
        lines.append(f"Baseline Top-1 Chunk ID: {baseline_results[0]['chunk_id']}")
        lines.append(f"Reranked Top-1 Chunk ID: {reranked_results[0]['chunk_id']}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Corpus1 reranking completed. Output saved.")


if __name__ == "__main__":
    main()
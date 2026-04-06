import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama


# -----------------------------
# Paths
# -----------------------------
BASE_PATH = Path("/home/surendran/LangChain/ragCORE")

MODEL_DIR = BASE_PATH / "ragCORE_2/models/corpus1_contrastive_model"

INDEX_PATH = BASE_PATH / "vector_store/Modified/Corpus_Narrative_chunk75_revised/faiss_index.bin"
MAPPING_PATH = BASE_PATH / "vector_store/Modified/Corpus_Narrative_chunk75_revised/id_to_chunk.json"
QUERY_FILE = BASE_PATH / "Data/query_set_v1.txt"

OUTPUT_FILE = BASE_PATH / "Output/retrieval_generation_contrastive_corpus1.txt"


# -----------------------------
# Model loaders
# -----------------------------
def load_embedding_model():
    return SentenceTransformer(str(MODEL_DIR))


def load_llm():
    return ChatOllama(model="llama3.2:3b", temperature=0.0)


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
# Strict grounded generation
# -----------------------------
def generate_answer(llm, query, context):
    prompt = f"""
You MUST answer ONLY using the provided context.

STRICT RULES:
- Do NOT add external knowledge
- Do NOT infer beyond the context
- Do NOT complete missing information from general knowledge
- If the answer is not explicitly available in the context, respond EXACTLY: Insufficient information

Context:
{context}

Question:
{query}

Answer:
"""
    response = llm.invoke(prompt)
    return response.content.strip()


# -----------------------------
# Main
# -----------------------------
def main():
    index = faiss.read_index(str(INDEX_PATH))

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        id_to_chunk = json.load(f)

    model = load_embedding_model()
    llm = load_llm()
    queries = load_queries(QUERY_FILE)

    print(f"Loaded contrastive model from: {MODEL_DIR}")
    print(f"Loaded FAISS index from: {INDEX_PATH}")
    print(f"FAISS index dimension: {index.d}")
    print("FAISS metric type: IP")

    results_lines = []

    high_count = 0
    mid_count = 0
    low_count = 0
    gaps = []

    for q_idx, query in enumerate(queries):
        results_lines.append("\n==============================")
        results_lines.append(f"Query {q_idx + 1}: {query}")
        results_lines.append("------------------------------")

        # Encode query using trained contrastive model
        q_emb = model.encode([query], convert_to_numpy=True).astype("float32")

        # Dimension check
        if q_emb.shape[1] != index.d:
            raise ValueError(
                f"Embedding dimension mismatch for query {q_idx + 1}: "
                f"query dim = {q_emb.shape[1]}, index dim = {index.d}"
            )

        # Normalize for IP search
        faiss.normalize_L2(q_emb)

        # Search
        scores, indices = index.search(q_emb, 3)

        top_scores = []
        top_chunks = []

        for rank, (idx, raw_score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:
                continue

            chunk = id_to_chunk.get(str(idx), "[Chunk not found in mapping]")
            score = float(raw_score)   # For normalized IP index, raw_score is the usable similarity
            band = get_band(score)

            top_scores.append(score)
            top_chunks.append(chunk)

            results_lines.append(
                f"Rank {rank + 1} | Raw: {raw_score:.4f} | Score: {score:.4f} | Band: {band}\n"
                f"Chunk: {chunk}\n"
            )

        # Top-1 summary count
        llm_response = ""
        if top_scores:
            top1_score = top_scores[0]
            top1_band = get_band(top1_score)
            top1_chunk = top_chunks[0]

            if top1_band == "HIGH":
                high_count += 1
                llm_response = generate_answer(llm, query, top1_chunk)
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

        # LLM response
        results_lines.append(f"LLM Response: {llm_response}")

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
    results_lines.append("FAISS metric type used: IP")
    results_lines.append(f"FAISS index dimension: {index.d}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(results_lines))

    print(f"Done. Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
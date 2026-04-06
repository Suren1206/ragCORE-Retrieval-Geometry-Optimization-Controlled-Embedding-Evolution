import json
import numpy as np
import faiss
import csv
from pathlib import Path


BASE_PATH = Path("/home/surendran/LangChain/ragCORE")

INDEX_PATH = BASE_PATH / "vector_store/Modified/Corpus_Narrative_chunk75/faiss_index.bin"
MAPPING_PATH = BASE_PATH / "vector_store/Modified/Corpus_Narrative_chunk75/id_to_chunk.json"
QUERY_FILE = BASE_PATH / "Data/query_set_v1.txt"
OUTPUT_FILE = BASE_PATH / "Output/retrieval_generate_Corpus1_chunk75.csv"


def load_embedding_model():
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model="nomic-embed-text")


def load_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(model="llama3.2:3b", temperature=0.0)


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


def detect_metric(index):
    metric_type = getattr(index, "metric_type", None)

    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return "IP"
    elif metric_type == faiss.METRIC_L2:
        return "L2"
    else:
        return "UNKNOWN"


def convert_to_score(raw_value, metric_kind):
    if metric_kind == "IP":
        return float(raw_value)
    elif metric_kind == "L2":
        return float(1 - (raw_value / 2))
    else:
        return float(raw_value)


def generate_answer(llm, query, context):
    prompt = f"""
You MUST answer ONLY using the provided context.

STRICT RULES:
- Do NOT add external knowledge
- Do NOT infer or complete missing info
- If not explicitly present, respond EXACTLY: Insufficient information

Context:
{context}

Question:
{query}

Answer:
"""
    response = llm.invoke(prompt)
    return response.content.strip()


def main():
    index = faiss.read_index(str(INDEX_PATH))

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        id_to_chunk = json.load(f)

    embed_model = load_embedding_model()
    llm = load_llm()
    queries = load_queries(QUERY_FILE)

    metric_kind = detect_metric(index)

    print(f"Loaded FAISS index from: {INDEX_PATH}")
    print(f"FAISS index dimension: {index.d}")
    print(f"FAISS metric type: {metric_kind}")

    rows = []

    for q_idx, query in enumerate(queries):
        q_emb = embed_model.embed_query(query)
        q_emb = np.array([q_emb], dtype="float32")

        if q_emb.shape[1] != index.d:
            raise ValueError(
                f"Embedding dimension mismatch for query {q_idx + 1}: "
                f"query dim = {q_emb.shape[1]}, index dim = {index.d}"
            )

        norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
        if np.any(norm == 0):
            raise ValueError(f"Zero vector encountered for query {q_idx + 1}: {query}")

        q_emb = q_emb / norm

        distances, indices = index.search(q_emb, 3)

        top_scores = []
        top1_chunk = ""
        generated_answer = ""

        for rank, (idx, raw_value) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:
                continue

            chunk = id_to_chunk.get(str(idx), "[Chunk not found in mapping]")
            score = convert_to_score(raw_value, metric_kind)
            band = get_band(score)

            top_scores.append(score)

            if rank == 0:
                top1_chunk = chunk
                top1_band = band

            rows.append({
                "query_id": q_idx + 1,
                "query": query,
                "rank": rank + 1,
                "raw_value": round(float(raw_value), 4),
                "score": round(score, 4),
                "band": band,
                "score_gap": "",
                "chunk": chunk,
                "generated_answer": ""
            })

        if top_scores and len(top_scores) >= 2:
            score_gap = round(top_scores[0] - top_scores[1], 4)
        else:
            score_gap = ""

        if top_scores:
            if top1_band == "HIGH":
                generated_answer = generate_answer(llm, query, top1_chunk)
            else:
                generated_answer = ""

        for row in reversed(rows):
            if row["query_id"] == q_idx + 1 and row["rank"] == 1:
                row["score_gap"] = score_gap
                row["generated_answer"] = generated_answer
                break

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
                "chunk",
                "generated_answer"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Corpus1 CSV with grounded generation saved.")


if __name__ == "__main__":
    main()
import json
import re
import numpy as np
import faiss
from pathlib import Path
from collections import Counter
from itertools import combinations


# ----------------------------------------------------------------------------------------------
# Paths  : Same codes can be used for geometric profiles for other two corpus also for comparison
# ----------------------------------------------------------------------------------------------


BASE_PATH = Path("/home/surendran/LangChain/ragCORE")

VECTOR_STORE = BASE_PATH / "vector_store/Baseline/Corpus_Narrative_chunk50"
DATA_FILE = BASE_PATH / "Data/Corpus1_Narrative.txt"
QUERY_FILE = BASE_PATH / "Data/query_set_v1.txt"

INDEX_PATH = VECTOR_STORE / "faiss_index.bin"
MAPPING_PATH = VECTOR_STORE / "id_to_chunk.json"

OUTPUT_FILE = BASE_PATH / "Output/geometry_profile_corpus1.json"


# -----------------------------
# Model
# -----------------------------
def load_model():
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model="nomic-embed-text")


# -----------------------------
# Helpers
# -----------------------------
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def get_band(score):
    if score >= 0.75:
        return "HIGH"
    elif score >= 0.60:
        return "MID"
    else:
        return "LOW"


# -----------------------------
# 1. Vocabulary Entropy
# -----------------------------
def compute_entropy(text):
    tokens = normalize(text)
    counts = Counter(tokens)
    total = len(tokens)

    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * np.log(p)

    return float(entropy)


# -----------------------------
# 2. Avg Chunk Length
# -----------------------------
def compute_avg_chunk_len(chunks):
    lengths = [len(chunk.split()) for chunk in chunks]
    return sum(lengths) / len(lengths)


# -----------------------------
# 3. Embedding Variance
# -----------------------------
def compute_embedding_variance(emb):
    dim_var = np.var(emb, axis=0)
    return float(np.mean(dim_var))


# -----------------------------
# 4. Inter-chunk Similarity
# -----------------------------
def compute_interchunk_similarity(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / norms

    sims = []
    for i, j in combinations(range(len(emb_norm)), 2):
        sims.append(np.dot(emb_norm[i], emb_norm[j]))

    return float(np.mean(sims))


# -----------------------------
# 5. Score Gap Analysis
# -----------------------------
def compute_score_gaps(index, model, queries):
    high_gaps = []
    mid_gaps = []

    for q in queries:
        q_emb = model.embed_query(q)
        q_emb = np.array([q_emb]).astype("float32")
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        distances, _ = index.search(q_emb, 2)

        score1 = 1 - (distances[0][0] / 2)
        score2 = 1 - (distances[0][1] / 2)

        gap = score1 - score2
        band = get_band(score1)

        if band == "HIGH":
            high_gaps.append(gap)
        elif band == "MID":
            mid_gaps.append(gap)

    avg_high_gap = sum(high_gaps) / len(high_gaps) if high_gaps else 0
    avg_mid_gap = sum(mid_gaps) / len(mid_gaps) if mid_gaps else 0

    return avg_high_gap, avg_mid_gap


# -----------------------------
# Main
# -----------------------------
def main():

    # Load data
    text = DATA_FILE.read_text(encoding="utf-8")

    with open(MAPPING_PATH, "r") as f:
        id_to_chunk = json.load(f)

    chunks = list(id_to_chunk.values())

    # Load model + index
    model = load_model()
    index = faiss.read_index(str(INDEX_PATH))

    # Recreate embeddings (important for consistency)
    emb = model.embed_documents(chunks)
    emb = np.array(emb).astype("float32")

    # Load queries
    queries = [line.strip() for line in open(QUERY_FILE) if line.strip()]

    # -----------------------------
    # Compute metrics
    # -----------------------------
    vocab_entropy = compute_entropy(text)
    avg_chunk_len = compute_avg_chunk_len(chunks)
    emb_variance = compute_embedding_variance(emb)
    inter_sim = compute_interchunk_similarity(emb)

    high_gap, mid_gap = compute_score_gaps(index, model, queries)

    # -----------------------------
    # Output
    # -----------------------------
    
    results = {
    "vocabulary_entropy": float(vocab_entropy),
    "avg_chunk_length": float(avg_chunk_len),
    "embedding_variance_mean": float(emb_variance),
    "inter_chunk_similarity_mean": float(inter_sim),
    "rank_gap": {
        "avg_gap_high": float(high_gap),
        "avg_gap_mid": float(mid_gap)
    }
}

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print("Geometry profiling completed.")


if __name__ == "__main__":
    main()
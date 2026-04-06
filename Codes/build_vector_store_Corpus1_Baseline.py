import json
import numpy as np
import faiss
from pathlib import Path


# ----------------------------------------------------------------------
# Paths - same code can be used for re-chunking at 75 words and 100 words
# -----------------------------------------------------------------------
DATA_PATH = Path("/home/surendran/LangChain/ragCORE/Data/Corpus1_Narrative.txt")
OUTPUT_DIR = Path("/home/surendran/LangChain/ragCORE/vector_store/Baseline/Corpus_Narrative_chunk50")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Chunking
# -----------------------------
def load_text(path):
    return path.read_text(encoding="utf-8")


def chunk_text(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += (chunk_size - overlap)

    return chunks


# -----------------------------
# Embedding Model
# -----------------------------
def load_model():
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model="nomic-embed-text")


# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    # 1. Load + chunk
    text = load_text(DATA_PATH)
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    # 2. Load model
    model = load_model()

    # 3. Generate embeddings
    embeddings = model.embed_documents(chunks)
    embeddings = np.array(embeddings, dtype="float32")

    print(f"Embeddings shape: {embeddings.shape}")

    # 4. Normalize embeddings
    faiss.normalize_L2(embeddings)

    # 5. Build FAISS index using inner product
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"FAISS index size: {index.ntotal}")
    print(f"FAISS index dimension: {index.d}")

    # 6. Save index
    faiss.write_index(index, str(OUTPUT_DIR / "faiss_index.bin"))

    # 7. Save mapping
    id_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}

    with open(OUTPUT_DIR / "id_to_chunk.json", "w", encoding="utf-8") as f:
        json.dump(id_to_chunk, f, indent=4)

    print("Pipeline complete. Files saved in vector_store.")


if __name__ == "__main__":
    main()
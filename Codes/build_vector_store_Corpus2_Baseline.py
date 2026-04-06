import json
import numpy as np
import faiss
from pathlib import Path


# ---------------------------------------------
# Paths - Chunking stragegy - Each atomic rule is a chunk
# ---------------------------------------------
BASE_PATH = Path("/home/surendran/LangChain/ragCORE")

DATA_PATH = BASE_PATH / "Data/Corpus2_Rule_based.txt"
OUTPUT_DIR = BASE_PATH / "vector_store/Baseline/Corpus2_Rule_based"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Chunking (RULE-BASED LOGIC) 
# -----------------------------
def chunk_text(text):
    lines = text.split("\n")
    chunks = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Skip headers (single word or no period)
        if "." not in line:
            continue

        # Keep only valid statements
        chunks.append(line)

    return chunks


# -----------------------------
# Model
# -----------------------------
def load_model():
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model="nomic-embed-text")


# -----------------------------
# Main
# -----------------------------
def main():
    text = DATA_PATH.read_text(encoding="utf-8")
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    model = load_model()
    embeddings = model.embed_documents(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(OUTPUT_DIR / "faiss_index.bin"))

    id_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}
    with open(OUTPUT_DIR / "id_to_chunk.json", "w") as f:
        json.dump(id_to_chunk, f, indent=4)

    print("Corpus2 vector store created.")


if __name__ == "__main__":
    main()
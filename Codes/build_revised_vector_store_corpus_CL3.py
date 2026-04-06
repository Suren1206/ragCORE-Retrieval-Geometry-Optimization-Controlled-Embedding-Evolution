import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer


BASE_PATH = Path("/home/surendran/LangChain/ragCORE")

ORIGINAL_MAPPING = BASE_PATH / "vector_store/Modified/Corpus_Narrative_chunk75/id_to_chunk.json"
MODEL_DIR = BASE_PATH / "models/corpus1_contrastive_model"

OUTPUT_DIR = BASE_PATH / "vector_store/Modified/Corpus_Narrative_chunk75_revised"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = OUTPUT_DIR / "faiss_index.bin"
MAPPING_FILE = OUTPUT_DIR / "id_to_chunk.json"
EMBED_FILE = OUTPUT_DIR / "embeddings.npy"


def main():
    with open(ORIGINAL_MAPPING, "r", encoding="utf-8") as f:
        id_to_chunk = json.load(f)

    ordered_ids = sorted(id_to_chunk.keys(), key=lambda x: int(x))
    chunks = [id_to_chunk[i] for i in ordered_ids]

    model = SentenceTransformer(str(MODEL_DIR))
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))

    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump({i: chunk for i, chunk in enumerate(chunks)}, f, indent=4, ensure_ascii=False)

    np.save(EMBED_FILE, embeddings)

    print(f"Revised vector store created at: {OUTPUT_DIR}")
    print(f"Chunks indexed: {index.ntotal}")

    print(f"Embedding shape: {embeddings.shape}")
    print(f"FAISS dim: {index.d}")


if __name__ == "__main__":
    main()
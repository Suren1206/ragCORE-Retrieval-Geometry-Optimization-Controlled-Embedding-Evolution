import json
import re
from pathlib import Path


BASE_PATH = Path("/home/surendran/LangChain/ragCORE")

INPUT_FILE = BASE_PATH / "Data/Contrastive learning inputs.txt"

OUTPUT_DIR = BASE_PATH / "Output/contrastive_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUTPUT_DIR / "train_triplets.jsonl"
EVAL_FILE = OUTPUT_DIR / "eval_cases.json"


def clean_chunk(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^Chunk:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def parse_blocks(raw_text: str):
    raw_text = raw_text.replace("\r\n", "\n")
    blocks = re.split(r"\n#{10,}\n", raw_text)
    return [b.strip() for b in blocks if "Query" in b]


def parse_query_block(block: str):
    query_match = re.search(r"\(\d+\)\s*Query\s*(\d+)\s*:\s*(.+)", block)
    if not query_match:
        return None

    query_id = int(query_match.group(1))
    query_text = query_match.group(2).strip()

    positive_match = re.search(
        r"Positive Chunk\s*:\s*(.*?)(?=\n\s*Negative Chunk|\n\s*Negative chunk|\Z)",
        block,
        flags=re.DOTALL | re.IGNORECASE,
    )

    negative_matches = re.findall(
        r"Negative Chunk\s*\d*\s*:\s*(.*?)(?=\n\s*Negative Chunk|\n\s*Negative chunk|\Z)",
        block,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not negative_matches:
        negative_matches = re.findall(
            r"Negative chunk\s*\d*\s*:\s*(.*?)(?=\n\s*Negative Chunk|\n\s*Negative chunk|\Z)",
            block,
            flags=re.DOTALL | re.IGNORECASE,
        )

    positive_chunk = clean_chunk(positive_match.group(1)) if positive_match else None
    negative_chunks = [clean_chunk(x) for x in negative_matches if clean_chunk(x)]

    return {
        "query_id": query_id,
        "query": query_text,
        "positive_chunk": positive_chunk,
        "negative_chunks": negative_chunks,
    }


def main():
    raw_text = INPUT_FILE.read_text(encoding="utf-8")
    blocks = parse_blocks(raw_text)

    parsed_cases = []
    train_cases = []

    for block in blocks:
        item = parse_query_block(block)
        if not item:
            continue

        parsed_cases.append(item)

        if item["positive_chunk"] and item["negative_chunks"]:
            for neg in item["negative_chunks"]:
                train_cases.append({
                    "query_id": item["query_id"],
                    "query": item["query"],
                    "positive": item["positive_chunk"],
                    "negative": neg,
                })

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for row in train_cases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(parsed_cases, f, indent=4, ensure_ascii=False)

    print(f"Parsed cases: {len(parsed_cases)}")
    print(f"Training triplets written: {len(train_cases)}")
    print(f"Train file: {TRAIN_FILE}")
    print(f"Eval file: {EVAL_FILE}")


if __name__ == "__main__":
    main()
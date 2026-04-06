import json
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.losses import TripletDistanceMetric


BASE_PATH = Path("/home/surendran/LangChain/ragCORE/ragCORE_2")

TRAIN_FILE = BASE_PATH / "contrastive_data/train_triplets.jsonl"
MODEL_OUTPUT_DIR = BASE_PATH / "models/corpus1_contrastive_model"


def load_triplets(path: Path):
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append(
                InputExample(
                    texts=[row["query"], row["positive"], row["negative"]]
                )
            )

    return examples


def main():
    train_examples = load_triplets(TRAIN_FILE)

    if not train_examples:
        raise ValueError("No training triplets found. Run parse_contrastive_corpus1.py first.")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=2
    )

    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=TripletDistanceMetric.COSINE,
        triplet_margin=0.25
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=12,
        warmup_steps=0,
        output_path=str(MODEL_OUTPUT_DIR),
        show_progress_bar=True
    )

    print(f"Trained model saved to: {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
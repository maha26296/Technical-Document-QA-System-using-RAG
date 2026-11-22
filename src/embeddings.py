import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# File paths
INPUT_FILE = Path("data/processed/chunks.jsonl")
OUTPUT_DIR = Path("data/processed")
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "chunks_metadata.json"

# Lightweight open-source embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print("Loading document chunks...")

    chunks = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    texts = [chunk["text"] for chunk in chunks]

    print("Total chunks:", len(texts))

    print("\nLoading embedding model:", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Save numerical vectors
    np.save(EMBEDDINGS_FILE, embeddings)

    # Save corresponding chunk metadata
    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=4)

    print("\nEmbedding generation completed!")
    print("Embedding shape:", embeddings.shape)
    print("Embedding dimension:", embeddings.shape[1])
    print("Saved embeddings:", EMBEDDINGS_FILE)
    print("Saved metadata:", METADATA_FILE)


if __name__ == "__main__":
    main()

import json
import numpy as np
import faiss
from pathlib import Path

# File paths
EMBEDDINGS_FILE = Path("data/processed/embeddings.npy")
METADATA_FILE = Path("data/processed/chunks_metadata.json")
VECTOR_DB_DIR = Path("vector_db")

INDEX_FILE = VECTOR_DB_DIR / "faiss_index.bin"


def main():
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")

    embeddings = np.load(EMBEDDINGS_FILE).astype("float32")

    print("Embedding shape:", embeddings.shape)

    # Create FAISS index using inner product
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    # Add document vectors to the index
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, str(INDEX_FILE))

    print("\nFAISS vector database created successfully!")
    print("Total vectors stored:", index.ntotal)
    print("Vector dimension:", dimension)
    print("Index saved at:", INDEX_FILE)


if __name__ == "__main__":
    main()

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# File paths
INDEX_FILE = Path("vector_db/faiss_index.bin")
METADATA_FILE = Path("data/processed/chunks_metadata.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def load_resources():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_FILE))

    print("Loading document metadata...")
    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, index, metadata


def retrieve_documents(question, model, index, metadata, top_k=TOP_K):
    # Convert user question into an embedding
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Search for the most relevant chunks
    scores, indices = index.search(question_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({
            "score": float(score),
            "document_name": metadata[idx]["document_name"],
            "page_number": metadata[idx]["page_number"],
            "chunk_id": metadata[idx]["chunk_id"],
            "text": metadata[idx]["text"]
        })

    return results


def main():
    model, index, metadata = load_resources()

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Please enter a valid question.")
        return

    results = retrieve_documents(question, model, index, metadata)

    print("\nRelevant Documents:\n")

    for number, result in enumerate(results, start=1):
        print(f"Result {number}")
        print("Similarity Score:", round(result["score"], 4))
        print("Document:", result["document_name"])
        print("Page:", result["page_number"])
        print("Chunk ID:", result["chunk_id"])
        print("Text:", result["text"][:500], "...")
        print("-" * 60)


if __name__ == "__main__":
    main()

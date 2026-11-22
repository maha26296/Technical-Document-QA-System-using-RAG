import json
from pathlib import Path

INPUT_FILE = Path("data/processed/extracted_documents.jsonl")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])

        if chunk_text.strip():
            chunks.append(chunk_text)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def main():
    all_chunks = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        for line in file:
            page_data = json.loads(line)

            text = page_data["text"]
            chunks = create_chunks(text)

            for index, chunk_text in enumerate(chunks, start=1):
                all_chunks.append({
                    "chunk_id": (
                        f'{page_data["document_name"]}_'
                        f'{page_data["page_number"]}_{index}'
                    ),
                    "document_name": page_data["document_name"],
                    "page_number": page_data["page_number"],
                    "text": chunk_text
                })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("Text chunking completed!")
    print("Total chunks created:", len(all_chunks))
    print("Chunk size:", CHUNK_SIZE, "words")
    print("Chunk overlap:", CHUNK_OVERLAP, "words")
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()

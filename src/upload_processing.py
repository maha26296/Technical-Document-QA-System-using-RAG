import fitz
import numpy as np
import faiss


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def create_chunks(text):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE

        chunk_text = " ".join(words[start:end])

        if chunk_text.strip():
            chunks.append(chunk_text)

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def process_uploaded_pdfs(uploaded_files, model):

    all_chunks = []

    for uploaded_file in uploaded_files:

        print(f"Processing uploaded PDF: {uploaded_file.name}")

        pdf_bytes = uploaded_file.getvalue()

        pdf = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if not text:
                continue

            chunks = create_chunks(text)

            for index, chunk_text in enumerate(chunks, start=1):

                all_chunks.append({
                    "chunk_id": (
                        f"{uploaded_file.name}_"
                        f"{page_number}_{index}"
                    ),
                    "document_name": uploaded_file.name,
                    "page_number": page_number,
                    "text": chunk_text
                })

        pdf.close()

    if not all_chunks:
        raise ValueError(
            "No readable text found in the uploaded PDF documents."
        )

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    print("Generating embeddings for uploaded documents...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    dimension = embeddings.shape[1]

    uploaded_index = faiss.IndexFlatIP(dimension)
    uploaded_index.add(embeddings)

    print("Uploaded PDF processing completed!")
    print("Total chunks:", len(all_chunks))

    return uploaded_index, all_chunks

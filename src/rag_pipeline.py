
from retrieval import load_resources, retrieve_documents
from generation import generate_answer


SIMILARITY_THRESHOLD = 0.30


def run_rag_pipeline(question, model, index, metadata):
    print("\nSearching relevant documents...")

    retrieved_chunks = retrieve_documents(
        question,
        model,
        index,
        metadata,
        top_k=5
    )

    if not retrieved_chunks:
        return (
            "I cannot find enough information in the provided documents.",
            []
        )
    

    best_score = retrieved_chunks[0]["score"]

    print("Best similarity score:", round(best_score, 4))

    if best_score < SIMILARITY_THRESHOLD:
        print("\nNo sufficiently relevant information found.")
        return (
            "I cannot find enough information in the provided documents.",
            []
        )

    answer = generate_answer(question, retrieved_chunks)
    return answer, retrieved_chunks


def main():
    print("=" * 60)
    print("TECHNICAL DOCUMENT QUESTION-ANSWERING SYSTEM")
    print("Powered by RAG + FAISS + Qwen 2.5")
    print("=" * 60)

    model, index, metadata = load_resources()

    print("\nSystem is ready!")

    while True:
        question = input(
            "\nEnter your question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("\nThank you for using the RAG system!")
            break

        if not question:
            print("Please enter a valid question.")
            continue

        answer, sources = run_rag_pipeline(
            question, model, index, metadata
        )

        if not sources:
            print(
                "\nAnswer: I cannot find enough information "
                "in the provided documents."
            )
            continue

        print("\n" + "=" * 60)
        print("GENERATED ANSWER")
        print("=" * 60)
        print(answer)

        print("\n" + "=" * 60)
        print("SOURCE REFERENCES")
        print("=" * 60)

        for source in sources:
            print(
                f"- {source['document_name']} | "
                f"Page: {source['page_number']} | "
                f"Similarity: {source['score']:.4f}"
            )


if __name__ == "__main__":
    main()

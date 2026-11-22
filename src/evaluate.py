import json
from pathlib import Path

from retrieval import load_resources, retrieve_documents
from generation import generate_answer


QUESTIONS_FILE = Path("evaluation/evaluation_questions.json")
RESULTS_FILE = Path("evaluation/evaluation_results.json")

SIMILARITY_THRESHOLD = 0.30


def main():
    print("Loading evaluation questions...")

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    questions = data["evaluation_questions"]

    model, index, metadata = load_resources()

    results = []

    for item in questions:
        question = item["question"]
        question_type = item["type"]

        print("\n" + "=" * 60)
        print("Question:", question)
        print("Type:", question_type)

        retrieved = retrieve_documents(
            question, model, index, metadata, top_k=5
        )

        best_score = (
            retrieved[0]["score"] if retrieved else 0.0
        )

        if not retrieved or best_score < SIMILARITY_THRESHOLD:
            answer = (
                "I cannot find enough information "
                "in the provided documents."
            )
            status = "Rejected - Low Relevance"
        else:
            answer = generate_answer(question, retrieved)
            status = "Answer Generated"

        print("Best similarity score:", round(best_score, 4))
        print("Status:", status)
        print("Answer:", answer)

        sources = [
            {
                "document": chunk["document_name"],
                "page": chunk["page_number"],
                "score": chunk["score"]
            }
            for chunk in retrieved
            if chunk["score"] >= SIMILARITY_THRESHOLD
        ]

        results.append({
            "id": item["id"],
            "question": question,
            "type": question_type,
            "similarity_score": best_score,
            "status": status,
            "answer": answer,
            "sources": sources
        })

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("\nEvaluation completed!")
    print("Total questions:", len(results))
    print("Results saved to:", RESULTS_FILE)


if __name__ == "__main__":
    main()

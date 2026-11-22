import ollama

MODEL_NAME = "qwen2.5:1.5b"

def generate_answer(question, retrieved_chunks):

    # Check whether relevant documents were retrieved
    if not retrieved_chunks:
        return "I cannot find enough information in the provided documents."

    # Prepare context from retrieved chunks
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"""
Document: {chunk['document_name']}
Page: {chunk['page_number']}
Content: {chunk['text']}
"""
        )

    context = "\n".join(context_parts)

    # Grounded prompt to reduce hallucinations
    
    prompt = f"""
You are a technical document question-answering assistant.

Answer the user's question using ONLY the information
provided in the retrieved context.

Follow these rules strictly:
1. Do not use outside knowledge.
2. Do not invent facts, statistics, or conclusions.
3. If the context does not contain enough evidence,
   say: "I cannot find enough information in the provided documents."
4. Give a direct, clear, and concise answer.
5. Use 3 to 5 sentences whenever possible.
6. Complete your answer fully. Do not end with unfinished sentences.
7. Do not add unrelated information.
8. Do not make claims that are not supported by the context.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""


    print("\nGenerating answer using Qwen LLM...\n")

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={"temperature": 0.1, "num_predict": 500}
    )

    return response["message"]["content"]


if __name__ == "__main__":
    test_question = "What is the main objective of the BEIR benchmark?"

    sample_chunks = [
        {
            "document_name": "BEIR_Research_Paper.pdf",
            "page_number": 3,
            "text": "BEIR aims to provide a one-stop zero-shot evaluation benchmark for all diverse retrieval tasks."
        }
    ]

    answer = generate_answer(test_question, sample_chunks)

    print("Question:", test_question)
    print("\nAnswer:\n", answer)

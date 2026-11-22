import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from retrieval import load_resources
from rag_pipeline import run_rag_pipeline

app = FastAPI(
    title="Technical Document QA API",
    description="RAG-based Question Answering API using FAISS and Qwen LLM",
    version="1.0.0"
)

model, index, metadata = load_resources()


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Technical Document QA API is running!"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        return {
            "answer": "Please enter a valid question.",
            "sources": []
        }

    answer, retrieved_chunks = run_rag_pipeline(
        question,
        model,
        index,
        metadata
    )

    sources = []

    for chunk in retrieved_chunks:
        sources.append({
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "similarity_score": round(chunk["score"], 4)
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }

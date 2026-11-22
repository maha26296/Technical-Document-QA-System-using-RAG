# Technical Document Question-Answering System using RAG

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) based Question-Answering System that allows users to ask questions about technical research papers in PDF format.

The system retrieves relevant information from the documents using semantic search and generates context-based answers using a lightweight open-source Large Language Model (LLM).

It is designed to provide accurate, document-grounded responses while reducing unsupported answers and hallucinations.

## Key Features

- PDF document ingestion and text extraction.
- Text chunking with configurable overlap.
- Semantic embeddings using Sentence Transformers.
- Vector similarity search using FAISS.
- Local answer generation using Qwen 2.5 (1.5B).
- Source references with document names and page numbers.
- Similarity threshold for rejecting low-relevance questions.
- Evaluation using supported and unsupported questions.
- Fully local RAG pipeline without paid LLM APIs.


## Research Documents

This project uses the following two research papers as its knowledge source.

### 1. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models

- **Document:** BEIR_Research_Paper.pdf
- **Objective:** To introduce a benchmark for evaluating information retrieval models across diverse retrieval tasks, particularly in zero-shot settings.
- **Focus:** Information retrieval, benchmarking, and zero-shot evaluation.
- **Key takeaway:** Retrieval models should be evaluated across diverse datasets and tasks rather than relying on a single dataset.

### 2. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

- **Document:** RAG_Research_Paper.pdf
- **Objective:** To introduce Retrieval-Augmented Generation, which combines a retriever with a language generator.
- **Focus:** External knowledge retrieval and context-based text generation.
- **Key takeaway:** Retrieved external documents can provide additional knowledge to a language model when generating answers.

### Document Comparison

BEIR focuses on evaluating information retrieval systems across different datasets and tasks. RAG focuses on combining document retrieval with language generation to answer knowledge-intensive questions.

In this project, the BEIR paper provides information about retrieval evaluation, while the RAG paper explains the retrieval-augmented generation methodology used in the application.


## System Architecture

The application follows a modular Retrieval-Augmented Generation (RAG) architecture.

```text
       Technical PDF Documents
                 |
                 v
          PDF Ingestion
                 |
                 v
          Text Extraction
                 |
                 v
           Text Chunking
       (500 words, 100 overlap)
                 |
                 v
       Embedding Generation
      (all-MiniLM-L6-v2)
                 |
                 v
          FAISS Vector Store
                 |
                 v
           User Question
                 |
                 v
       Query Embedding
                 |
                 v
       Semantic Retrieval
          (Top 5 Chunks)
                 |
                 v
       Similarity Threshold
                 |
                 v
        Retrieved Context
                 |
                 v
       Qwen 2.5 (1.5B LLM)
                 |
                 v
        Generated Answer
                 |
                 v
      Source Document References
```

### Workflow Explanation

1. **Document Ingestion:** Extracts text from PDF files using PyMuPDF while preserving document names and page numbers.
2. **Text Chunking:** Splits extracted text into manageable chunks of 500 words with a 100-word overlap.
3. **Embedding Generation:** Converts text chunks into 384-dimensional vector representations using Sentence Transformers.
4. **Vector Storage:** Stores normalized embeddings in a FAISS index for efficient similarity search.
5. **Semantic Retrieval:** Converts the user's question into an embedding and retrieves the five most relevant chunks.
6. **Relevance Filtering:** Uses a similarity threshold of 0.30 to reject queries with insufficiently relevant results.
7. **Answer Generation:** Uses the Qwen 2.5 (1.5B) model through Ollama to generate answers from retrieved context.
8. **Source References:** Displays the source PDF names, page numbers, and similarity scores.


## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyMuPDF | Extracting text from PDF documents |
| Sentence Transformers | Generating semantic embeddings |
| all-MiniLM-L6-v2 | Lightweight embedding model |
| FAISS | Vector storage and similarity search |
| Ollama | Running the local LLM |
| Qwen 2.5 (1.5B) | Generating document-based answers |
| NumPy | Numerical operations and embedding storage |
| JSON / JSONL | Storing extracted text and metadata |
| Git & GitHub | Version control and project hosting |

## Technical Configuration

| Parameter | Configuration |
|---|---|
| Embedding Model | all-MiniLM-L6-v2 |
| Embedding Dimension | 384 |
| Chunk Size | 500 words |
| Chunk Overlap | 100 words |
| Vector Database | FAISS |
| Similarity Method | Cosine similarity |
| Retrieval Top-K | 5 |
| Similarity Threshold | 0.30 |
| LLM | Qwen 2.5 (1.5B) |
| LLM Temperature | 0.1 |
| Maximum Generated Tokens | 500 |
| Execution Device | CPU |

### Design Decisions

- **Lightweight Embedding Model:** all-MiniLM-L6-v2 provides compact 384-dimensional embeddings suitable for CPU-based execution.
- **FAISS:** IndexFlatIP performs exact inner-product search. Since embeddings are normalized, the inner product represents cosine similarity.
- **Qwen 2.5 (1.5B):** Selected as a lightweight open-source model that can run locally through Ollama.
- **Similarity Threshold:** A threshold of 0.30 is used as an initial relevance filter to prevent low-relevance questions from reaching the LLM.


## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/maha26296/Technical-Document-QA-System-using-RAG.git

cd Technical-Document-QA-System-using-RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare PDF Documents

Place the following research papers inside:

```text
data/raw/documents/
```

- BEIR_Research_Paper.pdf
- RAG_Research_Paper.pdf

### 5. Build the RAG Knowledge Base

Run the following commands in order:

```bash
python src/ingestion.py
python src/chunking.py
python src/embeddings.py
python src/vector_store.py
```

These steps extract PDF text, create chunks, generate embeddings, and build the FAISS vector index.

### 6. Install and Configure Ollama

Download and install Ollama from:

https://ollama.com/

Download the Qwen model:

```bash
ollama pull qwen2.5:1.5b
```

Ensure Ollama is running before starting answer generation.

### 7. Run the Application

```bash
python src/rag_pipeline.py
```

Enter a question about the research papers.

Type `exit` to close the application.

### 8. Run Evaluation

```bash
python src/evaluate.py
```

Evaluation results are saved in:

```text
evaluation/evaluation_results.json
```


## Evaluation

The system was tested using five manually selected questions.

### Evaluation Test Cases

| Test Case | Question | Expected Behaviour |
|---|---|---|
| 1 | What is the main objective of the BEIR benchmark? | Generate a document-based answer |
| 2 | What does zero-shot evaluation mean in information retrieval? | Generate a document-based answer |
| 3 | What is the main objective of RAG? | Generate a document-based answer |
| 4 | How does RAG use retrieved documents to generate answers? | Generate a document-based answer |
| 5 | What is the capital of Japan? | Reject unsupported question |

### Evaluation Summary

- Total questions tested: 5
- Supported questions: 4
- Unsupported questions: 1
- Supported questions generated answers: 4
- Unsupported question rejected: 1

### Hallucination Handling

The system was tested with an unsupported question:

**Question:** What is the capital of Japan?

**System Response:**

"I cannot find enough information in the provided documents."

The system rejected this question because its best similarity score was below the configured threshold of 0.30.

### Limitations

- Evaluation currently uses only five manually selected questions.
- The similarity threshold requires further calibration using a larger dataset.
- PDF extraction may contain formatting and character-encoding issues.
- Lightweight LLMs may occasionally generate incomplete or unsupported statements.
- Formal retrieval metrics such as Precision@K and Recall@K have not yet been calculated.

The current results demonstrate functional behaviour on the selected test cases, not a comprehensive measurement of overall system accuracy.

Detailed results are available in `evaluation/evaluation_results.json`.


## How Can We Obtain the Best Answers from a Lightweight LLM?

A lightweight LLM has limited parameters and computational resources. Therefore, document preprocessing and retrieval quality play an important role in generating accurate answers.

The following techniques can improve answer quality:

1. **Clean PDF Text:** Remove unnecessary spaces, repeated headers, footers, page numbers, and extraction noise.
2. **Preserve Document Structure:** Retain document names, page numbers, headings, and relevant metadata.
3. **Effective Chunking:** Divide documents into meaningful chunks instead of passing entire documents to the LLM. Use suitable chunk sizes and overlap to preserve context.
4. **Quality Embeddings:** Use a suitable embedding model to represent document content and user questions semantically.
5. **Relevant Retrieval:** Retrieve only the most relevant chunks and avoid sending unrelated information to the LLM.
6. **Context Optimization:** Provide concise, relevant context to reduce unnecessary token usage.
7. **Clear Prompt Instructions:** Instruct the LLM to answer only from the retrieved context and reject questions that lack sufficient evidence.
8. **Relevance Filtering:** Apply a similarity threshold to reject queries that do not match the available documents.

In this project, we use 500-word chunks with a 100-word overlap, normalized embeddings, Top-5 retrieval, and a similarity threshold of 0.30.

These preprocessing and retrieval techniques help a lightweight model focus on relevant information, reduce unnecessary context, and produce more reliable document-grounded answers.


## Project Structure

```text
Technical-Document-QA-System-using-RAG/
│
├── data/
│   ├── raw/
│   │   └── documents/
│   │       ├── BEIR_Research_Paper.pdf
│   │       └── RAG_Research_Paper.pdf
│   └── processed/
│       ├── extracted_documents.jsonl
│       ├── chunks.jsonl
│       ├── embeddings.npy
│       └── chunks_metadata.json
│
├── src/
│   ├── download_dataset.py
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── rag_pipeline.py
│   └── evaluate.py
│
├── evaluation/
│   ├── evaluation_questions.json
│   ├── evaluation_results.json
│   └── evaluation_report.txt
│
├── vector_db/
│   └── faiss_index.bin
│
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: Processed files and the FAISS index are generated locally by running the build commands.*

## Conclusion

This project demonstrates the implementation of a complete Retrieval-Augmented Generation pipeline using open-source technologies.

By combining semantic search, FAISS vector retrieval, and the Qwen 2.5 (1.5B) language model, the application can answer questions based on technical PDF documents and display relevant source references.

The project also demonstrates how document preprocessing, effective chunking, relevant context retrieval, and prompt engineering can help improve the reliability of lightweight LLM-based applications.

## Future Enhancements

- Integrate larger document collections.
- Implement formal retrieval evaluation using Precision@K and Recall@K.
- Improve PDF text cleaning and extraction.
- Add a web-based user interface.
- Experiment with different embedding models and chunking strategies.
- Evaluate additional lightweight open-source LLMs.

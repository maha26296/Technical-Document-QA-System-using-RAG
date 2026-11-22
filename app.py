import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from retrieval import load_resources
from rag_pipeline import run_rag_pipeline

st.set_page_config(
    page_title="Technical Document QA System",
    page_icon="📚",
    layout="centered"
)


@st.cache_resource
def initialize_rag():
    return load_resources()


st.title("📚 Technical Document Question Answering System")

st.write(
    "Ask questions about the research documents. "
    "The system retrieves relevant information and generates "
    "answers using a locally running Qwen LLM."
)

st.subheader("Available Research Documents")

st.write("1. BEIR Research Paper")
st.write("2. RAG Research Paper")

st.divider()

try:
    model, index, metadata = initialize_rag()
    st.success("Knowledge base loaded successfully!")
except Exception as error:
    st.error(f"Unable to load the knowledge base: {error}")
    st.stop()

st.subheader("Ask Your Question")

question = st.text_input(
    "Enter your question:",
    placeholder="What is the main objective of the BEIR benchmark?"
)

if st.button("Generate Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            try:
                answer, sources = run_rag_pipeline(
                    question,
                    model,
                    index,
                    metadata
                )

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Source References")

                if sources:
                    for source in sources:
                        st.write(
                            f"📄 {source['document_name']} "
                            f"— Page {source['page_number']} "
                            f"— Similarity: {source['score']:.4f}"
                        )
                else:
                    st.info("No relevant source documents found.")

            except Exception as error:
                st.error(f"An error occurred: {error}")

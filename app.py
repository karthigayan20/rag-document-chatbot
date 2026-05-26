"""
RAG Document Chatbot - Main Application
Author: Built with LangChain + ChromaDB + Groq
"""

import streamlit as st
import os
from dotenv import load_dotenv

from rag.document_processor import DocumentProcessor
from rag.vector_store import VectorStore
from rag.chain import RAGChain

load_dotenv()

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 10px 16px;
        text-align: center;
    }
    .source-box {
        background: #f8f9fa;
        border-left: 4px solid #4f8ef7;
        padding: 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📚 RAG Document Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your documents and have an intelligent conversation with them using AI</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get your free API key at console.groq.com",
        placeholder="gsk_..."
    )

    st.markdown("---")

    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drag & drop or click to upload",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Supports PDF and plain text files",
    )

    if uploaded_files and groq_api_key:
        if st.button("🔄 Process Documents", type="primary", use_container_width=True):
            with st.spinner("Extracting text and creating embeddings..."):
                try:
                    processor = DocumentProcessor()
                    all_chunks = []

                    progress = st.progress(0)
                    for i, file in enumerate(uploaded_files):
                        chunks = processor.process(file)
                        all_chunks.extend(chunks)
                        progress.progress((i + 1) / len(uploaded_files))

                    vector_store = VectorStore()
                    vector_store.create(all_chunks)
                    st.session_state.vector_store = vector_store

                    rag_chain = RAGChain(groq_api_key=groq_api_key)
                    st.session_state.rag_chain = rag_chain

                    st.session_state.doc_count = len(uploaded_files)
                    st.session_state.chunk_count = len(all_chunks)
                    st.session_state.messages = []

                    st.success(f"✅ Ready! {len(uploaded_files)} file(s) processed.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    elif uploaded_files and not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key above.")

    st.markdown("---")

    if st.session_state.vector_store:
        st.subheader("📊 Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Files", st.session_state.doc_count)
        with col2:
            st.metric("Chunks", st.session_state.chunk_count)

        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.rag_chain = None
            st.session_state.doc_count = 0
            st.session_state.chunk_count = 0
            st.rerun()

    st.markdown("---")
    st.markdown("**🔗 Tech Stack**")
    st.markdown("- 🧠 **LLM:** Groq (Llama 3)")
    st.markdown("- 🔢 **Embeddings:** MiniLM-L6")
    st.markdown("- 🗃️ **Vector DB:** ChromaDB")
    st.markdown("- 🖥️ **UI:** Streamlit")

# ── Main Chat Area ─────────────────────────────────────────────────────────────
if st.session_state.vector_store and st.session_state.rag_chain:

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📄 Source Chunks Used"):
                    for i, src in enumerate(message["sources"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.markdown(f'<div class="source-box">{src[:600]}{"..." if len(src) > 600 else ""}</div>', unsafe_allow_html=True)
                        if i < len(message["sources"]):
                            st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    retrieved_docs = st.session_state.vector_store.retrieve(prompt, k=4)
                    response, sources = st.session_state.rag_chain.generate(
                        question=prompt,
                        retrieved_docs=retrieved_docs
                    )
                    st.markdown(response)

                    if sources:
                        with st.expander("📄 Source Chunks Used"):
                            for i, src in enumerate(sources, 1):
                                st.markdown(f"**Chunk {i}:**")
                                st.markdown(f'<div class="source-box">{src[:600]}{"..." if len(src) > 600 else ""}</div>', unsafe_allow_html=True)
                                if i < len(sources):
                                    st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources,
                    })
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")

else:
    # Landing state
    st.info("👈 **Get started:** Enter your Groq API key and upload documents in the sidebar.")

    st.markdown("### How it works")
    col1, col2, col3, col4 = st.columns(4)
    steps = [
        ("1️⃣", "Upload", "PDF or TXT documents via the sidebar"),
        ("2️⃣", "Process", "Documents are split into chunks & embedded"),
        ("3️⃣", "Retrieve", "Relevant chunks are fetched using vector search"),
        ("4️⃣", "Answer", "Groq LLM generates a grounded, cited response"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], steps):
        with col:
            st.markdown(f"#### {icon} {title}")
            st.caption(desc)

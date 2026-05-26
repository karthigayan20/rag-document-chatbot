# 📚 RAG Document Chatbot

> **Chat intelligently with your documents using Retrieval-Augmented Generation (RAG)**

A production-style AI application that lets users upload PDF or TXT documents and ask natural-language questions about them. Built with LangChain, ChromaDB, Groq (Llama 3), and Streamlit.

---

## 🎬 Demo

Upload any PDF → Ask questions → Get grounded, cited answers

```
User: "What are the key risk factors mentioned in this report?"
Bot:  "According to Chunk 2 of the document, the key risk factors are:
       1. Market volatility due to...
       2. Regulatory changes in...
       3. Supply chain disruption..."
```

---

## ✨ Features

- 📄 **Multi-format support** — Upload PDF and TXT files
- 🔍 **Semantic search** — Finds the most relevant chunks, not just keyword matches
- 🧠 **Grounded answers** — LLM only uses document content, never hallucinates outside facts
- 📌 **Source transparency** — Every answer shows which chunks were used
- 🚀 **Fast inference** — Groq's LPU delivers sub-second LLM responses
- 💸 **Fully free** — Groq free tier + local embeddings = $0 to run
- 🖥️ **Clean UI** — Streamlit chat interface with session management

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         User Interface                        │
│                    (Streamlit Chat App)                       │
└──────────────────┬───────────────────────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │  Document Processor │  ← PyMuPDF / text decoder
         │  (Chunking / Split) │  ← RecursiveCharacterTextSplitter
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │    Vector Store     │  ← sentence-transformers (MiniLM-L6)
         │    (ChromaDB)       │  ← cosine similarity index
         └─────────┬──────────┘
                   │  top-k chunks
         ┌─────────▼──────────┐
         │     RAG Chain       │  ← LangChain prompt builder
         │  (Groq / Llama 3)   │  ← grounded answer generation
         └────────────────────┘
```

---

## 🛠️ Tech Stack

| Component        | Technology                          | Why                              |
|-----------------|-------------------------------------|----------------------------------|
| LLM             | Groq API (Llama 3 8B)              | Free, blazing fast inference     |
| Embeddings      | sentence-transformers/all-MiniLM-L6 | Free, no API key, runs locally   |
| Vector Database | ChromaDB (in-memory)                | Easy to set up, cosine similarity|
| Orchestration   | LangChain                           | Industry standard, modular       |
| PDF Parsing     | PyMuPDF (fitz)                      | Fast, accurate text extraction   |
| UI              | Streamlit                           | Rapid AI app development         |

---

## 📁 Project Structure

```
rag-document-chatbot/
├── app.py                      # Main Streamlit application
├── rag/
│   ├── __init__.py
│   ├── document_processor.py   # PDF/TXT loading and chunking
│   ├── vector_store.py         # ChromaDB + embedding management
│   └── chain.py                # LangChain RAG prompt + Groq LLM
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- A free [Groq API key](https://console.groq.com) (takes 2 minutes)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-document-chatbot.git
cd rag-document-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ First run downloads the MiniLM embedding model (~90 MB). This is a one-time download.

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

### 5. Run the application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🧪 Usage

1. **Enter your Groq API key** in the sidebar (or set it in `.env`)
2. **Upload one or more PDF/TXT files** using the file uploader
3. Click **"Process Documents"** — this chunks and embeds your files
4. **Type your question** in the chat box
5. The app retrieves the most relevant chunks and generates a grounded answer
6. Expand **"Source Chunks Used"** to see what the LLM referenced

---

## 🔑 Key Concepts Demonstrated

| Concept                    | Where                          |
|---------------------------|--------------------------------|
| Document chunking strategy | `rag/document_processor.py`    |
| Dense vector embeddings    | `rag/vector_store.py`          |
| Cosine similarity search   | `rag/vector_store.py`          |
| RAG prompt engineering     | `rag/chain.py`                 |
| Context-grounded generation| `rag/chain.py`                 |
| LLM integration (Groq)     | `rag/chain.py`                 |
| Stateful chat UI           | `app.py`                       |

---

## ⚙️ Configuration

You can tune the following in code:

| Parameter       | Default | File                       | Effect                          |
|----------------|---------|----------------------------|---------------------------------|
| `chunk_size`   | 1000    | `document_processor.py`    | Larger = more context per chunk |
| `chunk_overlap`| 200     | `document_processor.py`    | Prevents sentence cut-offs      |
| `k` (top-k)    | 4       | `vector_store.py`          | More chunks = more context      |
| `temperature`  | 0.1     | `chain.py`                 | Lower = more deterministic      |
| `model`        | llama3-8b-8192 | `chain.py`         | Groq model to use               |

---

## 📈 Possible Extensions

- [ ] Add support for `.docx` and `.csv` files
- [ ] Persist ChromaDB to disk for cross-session memory
- [ ] Add a reranker (Cohere / cross-encoder) for better retrieval
- [ ] Implement query reformulation for multi-turn conversations
- [ ] Add evaluation metrics (faithfulness, answer relevance)
- [ ] Deploy to Hugging Face Spaces

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [LangChain](https://langchain.com) — LLM orchestration framework
- [Groq](https://groq.com) — Ultra-fast LLM inference
- [ChromaDB](https://trychroma.com) — Open-source vector database
- [Sentence Transformers](https://sbert.net) — Semantic embedding models

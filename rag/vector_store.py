"""
Vector Store
Wraps ChromaDB with HuggingFace embeddings for similarity search.
No API key required – embeddings run locally via sentence-transformers.
"""

import uuid
from typing import List

import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStore:
    """
    In-memory ChromaDB vector store backed by sentence-transformers embeddings.
    Supports document ingestion and semantic similarity retrieval.
    """

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        # In-memory client (no disk persistence needed for demo)
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def create(self, documents: List[Document]) -> None:
        """
        Embed all document chunks and store them in a new ChromaDB collection.

        Args:
            documents: List of LangChain Document objects (pre-chunked)
        """
        collection_name = f"rag_{uuid.uuid4().hex[:10]}"
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [str(uuid.uuid4()) for _ in documents]
        embeddings = self.embedding_model.embed_documents(texts)

        # ChromaDB batch add
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

    def retrieve(self, query: str, k: int = 4) -> List[str]:
        """
        Retrieve the top-k most relevant chunks for a given query.

        Args:
            query: User's question string
            k:     Number of chunks to retrieve

        Returns:
            List of raw chunk texts
        """
        if not self.collection:
            raise RuntimeError(
                "Vector store is empty. Please process documents first."
            )

        # Clamp k to the number of stored docs
        k = min(k, self.collection.count())
        if k == 0:
            return []

        query_embedding = self.embedding_model.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        return results["documents"][0] if results["documents"] else []

    @property
    def is_ready(self) -> bool:
        return self.collection is not None and self.collection.count() > 0

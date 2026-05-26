"""
Document Processor
Handles loading and chunking of uploaded documents (PDF / TXT).
"""

import io
from typing import List

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class DocumentProcessor:
    """
    Loads raw files and returns a list of LangChain Document chunks
    ready for embedding.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process(self, uploaded_file) -> List[Document]:
        """
        Main entry point. Detects file type and returns a list of chunks.

        Args:
            uploaded_file: Streamlit UploadedFile object

        Returns:
            List of LangChain Document objects (chunks)
        """
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".pdf"):
            raw_text = self._extract_pdf(file_bytes)
        elif file_name.endswith(".txt"):
            raw_text = self._extract_txt(file_bytes)
        else:
            raise ValueError(
                f"Unsupported file type: '{uploaded_file.name}'. "
                "Please upload a PDF or TXT file."
            )

        if not raw_text.strip():
            raise ValueError(
                f"Could not extract any text from '{uploaded_file.name}'. "
                "The file may be empty or image-based."
            )

        chunks = self.splitter.create_documents(
            texts=[raw_text],
            metadatas=[{"source": uploaded_file.name}],
        )
        return chunks

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_pdf(self, file_bytes: bytes) -> str:
        """Extract plain text from a PDF using PyMuPDF."""
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = []
        for page in pdf_doc:
            pages_text.append(page.get_text("text"))
        pdf_doc.close()
        return "\n\n".join(pages_text)

    def _extract_txt(self, file_bytes: bytes) -> str:
        """Decode a plain-text file; tries UTF-8, falls back to latin-1."""
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1")

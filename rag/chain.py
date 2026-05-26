"""
RAG Chain
Combines retrieved context with the user question and calls the Groq LLM.
"""

from typing import List, Tuple

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage


SYSTEM_PROMPT = """You are a precise and helpful document assistant.
Your job is to answer the user's question using ONLY the context chunks provided below.

Rules:
1. If the answer is clearly found in the context, answer it accurately and concisely.
2. If the answer is partially found, share what you know and state what is missing.
3. If the answer is NOT found in the context at all, say:
   "I couldn't find that information in the uploaded documents."
4. Never make up facts. Never use outside knowledge.
5. When possible, mention which part of the context supports your answer.
"""


class RAGChain:
    """
    Retrieval-Augmented Generation chain using Groq as the LLM backend.
    """

    def __init__(
        self,
        groq_api_key: str,
        model: str = "llama3-8b-8192",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate(
        self,
        question: str,
        retrieved_docs: List[str],
    ) -> Tuple[str, List[str]]:
        """
        Build a prompt from retrieved chunks and generate a grounded answer.

        Args:
            question:       The user's question
            retrieved_docs: List of relevant document chunks

        Returns:
            (answer_text, source_chunks)
        """
        if not retrieved_docs:
            return (
                "I couldn't find any relevant information in the uploaded documents "
                "for your question. Please try rephrasing or upload a more relevant document.",
                [],
            )

        # Format context block
        context_block = "\n\n---\n\n".join(
            f"[Chunk {i + 1}]:\n{chunk}"
            for i, chunk in enumerate(retrieved_docs)
        )

        user_prompt = f"""Context from the uploaded documents:

{context_block}

---

User Question: {question}

Answer:"""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        return response.content, retrieved_docs

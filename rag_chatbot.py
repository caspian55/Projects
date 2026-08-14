from typing import List, Dict, Any, Iterator, Optional
import ollama
from vector_store import VectorStore
SYSTEM_PROMPT = """You are a careful, helpful assistant that answers questions using ONLY the \
information provided in the "Context" section below.

Rules:
- Base your answer strictly on the provided context.
- If the context does not contain enough information to answer, say so clearly \
instead of guessing or using outside knowledge.
- Be concise and directly answer the question first, then add supporting detail if useful.
- When helpful, mention which source document the information came from.
"""
class RAGChatbot:
    def __init__(
        self,
        vector_store: VectorStore,
        llm_model: str = "llama3.1",
        n_results: int = 4,
    ):
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.n_results = n_results
        self.last_sources: List[Dict[str, Any]] = []

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        return self.vector_store.query(query, n_results=self.n_results)

    def _build_messages(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        if context_chunks:
            context_text = "\n\n---\n\n".join(
                f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
            )
        else:
            context_text = "(No relevant content was found in the uploaded documents.)"

        user_content = (
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context above. If the context doesn't contain the answer, "
            "say that clearly."
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if chat_history:
            # Keep a short rolling window so prompts don't grow unbounded
            messages.extend(chat_history[-6:])

        messages.append({"role": "user", "content": user_content})
        return messages

    def stream_response(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[str]:
        """
        Retrieve context, call the LLM in streaming mode, and yield text tokens
        as they arrive. After the generator is exhausted, `self.last_sources`
        holds the chunks that were used, for citation display.
        """
        context_chunks = self.retrieve(query)
        self.last_sources = context_chunks

        messages = self._build_messages(query, context_chunks, chat_history)

        stream = ollama.chat(model=self.llm_model, messages=messages, stream=True)
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    def get_response(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Non-streaming convenience method, mainly useful for testing/scripting."""
        context_chunks = self.retrieve(query)
        self.last_sources = context_chunks
        messages = self._build_messages(query, context_chunks, chat_history)
        response = ollama.chat(model=self.llm_model, messages=messages, stream=False)
        return response["message"]["content"]
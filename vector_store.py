from typing import List, Dict, Any
import ollama
import chromadb
from chromadb import EmbeddingFunction
class OllamaEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible embedding function backed by a local Ollama model."""

    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings

    def name(self) -> str:
        return f"ollama-{self.model_name}"

    @staticmethod
    def build_from_config(config: dict) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction(model_name=config.get("model_name", "nomic-embed-text"))

    def get_config(self) -> dict:
        return {"model_name": self.model_name}


class VectorStore:
    """In-memory Chroma collection scoped to a single chat session."""

    def __init__(self, embedding_model: str = "nomic-embed-text", collection_name: str = "documents"):
        self.client = chromadb.EphemeralClient()
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.embedding_fn = OllamaEmbeddingFunction(embedding_model)
        self._create_collection()

    def _create_collection(self) -> None:
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: List[str], source: str) -> None:
        """Embed and store a list of text chunks under a given source filename."""
        if not chunks:
            return

        start = self.collection.count()
        ids = [f"{source}::{start + i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

        # Chroma batches internally, but we chunk the insert ourselves too,
        # to keep individual embedding calls to Ollama small and resilient.
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            self.collection.add(
                documents=chunks[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size],
            )

    def query(self, query_text: str, n_results: int = 4) -> List[Dict[str, Any]]:
        """Return the top-n most relevant chunks for a query."""
        count = self.collection.count()
        if count == 0:
            return []

        n_results = min(n_results, count)
        results = self.collection.query(query_texts=[query_text], n_results=n_results)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        return [
            {"text": doc, "source": meta.get("source", "unknown"), "distance": dist}
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def reset(self) -> None:
        """Delete all indexed documents."""
        self.client.delete_collection(self.collection_name)
        self._create_collection()

    def document_count(self) -> int:
        return self.collection.count()
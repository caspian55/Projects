import random
import sys
import types
# ---- Mock the `ollama` module before anything imports it ----
mock_ollama = types.ModuleType("ollama")
def fake_embeddings(model, prompt):
    random.seed(hash(prompt) % (2**32))
    return {"embedding": [random.random() for _ in range(32)]}
def fake_chat(model, messages, stream=False):
    user_msg = messages[-1]["content"]
    reply = f"[mock answer from {model}] I found relevant info and answered: {user_msg[:40]}..."
    if stream:
        def gen():
            for word in reply.split(" "):
                yield {"message": {"content": word + " "}}
        return gen()
    return {"message": {"content": reply}}
def fake_list():
    return {"models": [{"model": "llama3.1"}, {"model": "nomic-embed-text"}]}
mock_ollama.embeddings = fake_embeddings
mock_ollama.chat = fake_chat
mock_ollama.list = fake_list
sys.modules["ollama"] = mock_ollama

# ---- Now import the real project modules ----
from document_loader import chunk_text, load_document
from vector_store import VectorStore
from rag_chatbot import RAGChatbot

# 1. Chunking test
sample_text = (
    "Ollama lets you run large language models locally.\n\n"
    "Retrieval-Augmented Generation (RAG) combines a retriever with a generator. "
    "It first finds relevant chunks of text from a knowledge base, then feeds them "
    "to an LLM as context.\n\n"
    "This project shows how to build a simple, local RAG chatbot using Streamlit, "
    "ChromaDB, and Ollama, without needing any cloud APIs or API keys.\n\n"
) * 5
chunks = chunk_text(sample_text, chunk_size=200, chunk_overlap=40)
assert len(chunks) > 1, "Expected multiple chunks"
for c in chunks:
    assert len(c) <= 260, f"Chunk too long: {len(c)}"
print(f"[OK] chunk_text produced {len(chunks)} chunks, e.g.: {chunks[0][:60]!r}")

# 2. Vector store test
vs = VectorStore(embedding_model="nomic-embed-text", collection_name="test_collection")
vs.add_documents(chunks, source="test_doc.txt")
assert vs.document_count() == len(chunks)
print(f"[OK] VectorStore indexed {vs.document_count()} chunks")

results = vs.query("What is RAG?", n_results=3)
assert len(results) == 3
assert all("text" in r and "source" in r and "distance" in r for r in results)
print(f"[OK] VectorStore.query returned {len(results)} results, top source={results[0]['source']}")

vs.reset()
assert vs.document_count() == 0
print("[OK] VectorStore.reset() cleared the collection")

vs.add_documents(chunks, source="test_doc.txt")

# 3. RAGChatbot test (streaming)
bot = RAGChatbot(vector_store=vs, llm_model="llama3.1", n_results=2)
stream_output = "".join(bot.stream_response("What does this project demonstrate?"))
assert len(stream_output) > 0
assert len(bot.last_sources) == 2
print(f"[OK] RAGChatbot.stream_response produced {len(stream_output)} chars, "
      f"used {len(bot.last_sources)} source chunks")

# 4. RAGChatbot test (non-streaming) + chat history
reply = bot.get_response("Follow up question", chat_history=[
    {"role": "user", "content": "What does this project demonstrate?"},
    {"role": "assistant", "content": stream_output},
])
assert isinstance(reply, str) and len(reply) > 0
print(f"[OK] RAGChatbot.get_response with history -> {reply[:60]!r}")

# 5. Document loader dispatch test (unsupported extension)
try:
    load_document("/tmp/fakefile.xyz")
    raise SystemExit("Expected ValueError for unsupported extension")
except ValueError as e:
    print(f"[OK] load_document correctly rejected unsupported extension: {e}")

print("\nALL SMOKE TESTS PASSED")
import os
import sys
import tempfile
from pathlib import Path
_APP_DIR = str(Path(__file__).resolve().parent)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import ollama
import streamlit as st
try:
    from document_loader import load_document, chunk_text
    from vector_store import VectorStore
    from rag_chatbot import RAGChatbot
except ModuleNotFoundError as e:
    st.error(
        f"Could not import a required local module: {e}\n\n"
        f"Make sure document_loader.py, vector_store.py, and rag_chatbot.py "
        f"are in the same folder as app.py:\n{_APP_DIR}"
    )
    st.stop()
st.set_page_config(page_title="Smart Document Q&A", page_icon="📚", layout="wide")
# Ollama helpers
def check_ollama_connection() -> bool:
    try:
        ollama.list()
        return True
    except Exception:
        return False
def get_available_models():
    try:
        response = ollama.list()
        models = response.get("models", [])
        names = []
        for m in models:
            # Different ollama-python versions expose the name under
            # different keys ("model" vs "name") - handle both.
            name = m.get("model") or m.get("name")
            if name:
                names.append(name)
        return sorted(set(names))
    except Exception:
        return []
# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

def init_state():
    defaults = {
        "messages": [],
        "vector_store": None,
        "chatbot": None,
        "processed_files": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
init_state()

st.title("📚 Smart Document Q&A Chatbot")
st.caption("Local RAG chatbot — your documents and questions never leave your machine (powered by Ollama).")

if not check_ollama_connection():
    st.error("⚠️ Can't connect to Ollama.")
    st.markdown(
        "Make sure Ollama is installed and running:\n\n"
        "1. Install it from **https://ollama.com/download**\n"
        "2. Start the server: `ollama serve`\n"
        "3. Pull a chat model and an embedding model, e.g.:\n"
        "   ```\n"
        "   ollama pull llama3.1\n"
        "   ollama pull nomic-embed-text\n"
        "   ```\n"
        "4. Refresh this page."
    )
    st.stop()

available_models = get_available_models()

if not available_models:
    st.warning(
        "Ollama is running, but no local models were found. "
        "Pull one first, e.g. `ollama pull llama3.1` and `ollama pull nomic-embed-text`."
    )
# --------------------------------------------------------------------------
# Sidebar: configuration + document upload
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    embed_hint_options = [m for m in available_models if "embed" in m.lower()]
    llm_hint_options = [m for m in available_models if "embed" not in m.lower()]

    llm_model = st.selectbox(
        "LLM model (for answers)",
        options=llm_hint_options or available_models or ["llama3.1"],
    )
    embedding_model = st.selectbox(
        "Embedding model (for retrieval)",
        options=embed_hint_options or available_models or ["nomic-embed-text"],
    )

    st.divider()
    chunk_size = st.slider("Chunk size (characters)", 200, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 500, 200, step=50)
    n_results = st.slider("Chunks retrieved per question", 1, 10, 4)

    st.divider()
    st.subheader("📄 Upload documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT, or MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    process_clicked = st.button(
        "🚀 Process documents", use_container_width=True, disabled=not uploaded_files
    )

    if process_clicked:
        if st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore(embedding_model=embedding_model)

        new_files = [
            f for f in uploaded_files if f.name not in st.session_state.processed_files
        ]

        if not new_files:
            st.info("All selected files have already been processed.")
        else:
            progress = st.progress(0.0, text="Starting...")
            for i, uploaded_file in enumerate(new_files):
                progress.progress(i / len(new_files), text=f"Processing {uploaded_file.name}...")

                suffix = os.path.splitext(uploaded_file.name)[1]
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    text = load_document(tmp_path)
                    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                    if not chunks:
                        st.warning(f"No usable text extracted from {uploaded_file.name}, skipping.")
                        continue

                    st.session_state.vector_store.add_documents(chunks, source=uploaded_file.name)
                    st.session_state.processed_files.append(uploaded_file.name)

                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            progress.progress(1.0, text="Done!")

            st.session_state.chatbot = RAGChatbot(
                vector_store=st.session_state.vector_store,
                llm_model=llm_model,
                n_results=n_results,
            )

            st.success(
                f"Indexed {len(st.session_state.processed_files)} document(s), "
                f"{st.session_state.vector_store.document_count()} chunks total."
            )

    if st.session_state.processed_files:
        st.divider()
        st.subheader("📚 Indexed documents")
        for fname in st.session_state.processed_files:
            st.text(f"✓ {fname}")

        if st.button("🗑️ Clear knowledge base", use_container_width=True):
            st.session_state.vector_store.reset()
            st.session_state.processed_files = []
            st.session_state.messages = []
            st.session_state.chatbot = None
            st.rerun()

# Keep an already-created chatbot's settings in sync with sidebar changes
if st.session_state.chatbot is not None:
    st.session_state.chatbot.llm_model = llm_model
    st.session_state.chatbot.n_results = n_results
# --------------------------------------------------------------------------
# Main chat interface
# --------------------------------------------------------------------------

def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📖 Sources ({len(sources)})"):
        for s in sources:
            relevance = max(0.0, 1 - s["distance"])
            st.markdown(f"**{s['source']}** · relevance {relevance:.2f}")
            preview = s["text"][:300] + ("..." if len(s["text"]) > 300 else "")
            st.caption(preview)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])
query = st.chat_input("Ask a question about your documents...")
if query:
    if st.session_state.chatbot is None:
        st.warning("Upload and process at least one document before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
            if m["role"] in ("user", "assistant")
        ]

        with st.chat_message("assistant"):
            try:
                response_text = st.write_stream(
                    st.session_state.chatbot.stream_response(query, chat_history)
                )
            except Exception as e:
                response_text = f"⚠️ Error generating a response: {e}"
                st.error(response_text)

            sources = st.session_state.chatbot.last_sources
            render_sources(sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text, "sources": sources}
        )
if not st.session_state.processed_files:
    st.info("👈 Upload one or more documents in the sidebar and click **Process documents** to get started.")
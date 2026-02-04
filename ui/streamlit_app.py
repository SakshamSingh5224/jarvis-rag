import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://api:8000").rstrip("/")

st.set_page_config(page_title="Jarvis-RAG", page_icon="🤖", layout="wide")

st.title("🤖 Jarvis-RAG Assistant")
st.caption("Self-hosted LLM (Ollama) + Pinecone RAG + Streamlit UI")

with st.sidebar:
    st.header("Knowledge Base")
    st.write("Upload files (PDF/TXT/MD) to ingest into Pinecone.")
    uploaded = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])
    ingest_btn = st.button("Ingest", use_container_width=True)

    st.divider()
    st.header("Settings")
    st.write(f"API: `{API_BASE}`")
    show_sources = st.toggle("Show sources", value=True)

def api_post(path, **kwargs):
    url = f"{API_BASE}{path}"
    r = requests.post(url, timeout=180, **kwargs)
    r.raise_for_status()
    return r.json()

# Ingest action
if ingest_btn and uploaded is not None:
    with st.spinner("Ingesting..."):
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        try:
            res = api_post("/ingest/upload", files=files)
            st.sidebar.success(f"Ingested `{uploaded.name}` (chunks: {res.get('chunks')})")
        except Exception as e:
            st.sidebar.error(f"Failed to ingest: {e}")

# Chat state
if "history" not in st.session_state:
    st.session_state.history = []  # list of {role, content}

# Display chat history
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Ask Jarvis...")
if user_msg:
    st.session_state.history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {"message": user_msg, "history": st.session_state.history[:-1]}
                res = api_post("/chat", json=payload)
                answer = res.get("answer", "")
                sources = res.get("sources", [])
                st.markdown(answer)
                if show_sources and sources:
                    st.caption("Sources: " + ", ".join([f"`{s}`" for s in sources]))
                st.session_state.history.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Chat failed: {e}")

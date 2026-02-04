# Jarvis-RAG (Self-hosted LLM + Pinecone + Chat UI)

A deployable "Build Your Own Jarvis" project:
- **Self-hosted LLM** via **Ollama** (runs locally/on your server)
- **Vector DB** via **Pinecone**
- **RAG** pipeline (ingest → chunk → embed → upsert; chat → retrieve → answer)
- **UI** via **Streamlit**
- **API** via **FastAPI**

> You can deploy this with Docker Compose. You provide your Pinecone API key and run Ollama.

---

## 1) Prerequisites

### Pinecone
1. Create a Pinecone account and an **index** (metric: cosine).
2. The index **dimension must match the embedding model** you use (default below is **1024** for `intfloat/e5-large-v2`).
3. Note your:
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX` (index name)
   - `PINECONE_NAMESPACE` (optional; defaults to `default`)

### Ollama (self-hosted LLM)
Install and run Ollama on the same machine (or reachable from Docker network).

Pull a model:
```bash
ollama pull llama3.1
```

Ollama runs at: `http://localhost:11434` by default.

---

## 2) Configure environment

Copy the example env file:
```bash
cp .env.example .env
```

Fill in:
- `PINECONE_API_KEY`
- `PINECONE_INDEX`
- `OLLAMA_MODEL` (e.g., `llama3.1`)

---

## 3) Run with Docker Compose

```bash
docker compose up --build
```

- API: http://localhost:8000
- UI:  http://localhost:8501

---

## 4) Use it

### Ingest documents
In the UI, upload PDFs/TXT/MD and click **Ingest**.

Or via API:
```bash
curl -X POST "http://localhost:8000/ingest/upload" \
  -F "file=@docs/sample.pdf"
```

### Chat
Ask questions in the UI, or:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"Summarize what you know about ..."}'
```

---

## 5) Notes

- If your Pinecone index dimension doesn't match your embedding model, ingestion will fail.
- Default embedding model: `intfloat/e5-large-v2` (1024 dims).
- You can swap embedding models via `EMBED_MODEL`. Update Pinecone index dimension accordingly.

---

## Project structure

```
jarvis-rag/
  backend/   # FastAPI + RAG + ingestion
  ui/        # Streamlit chat UI
  docker-compose.yml
  .env.example
```

---

## License
MIT

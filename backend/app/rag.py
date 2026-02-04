from __future__ import annotations

from typing import Any, Dict, List, Tuple
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from .settings import settings

SYSTEM_PROMPT = """You are Jarvis, a helpful enterprise AI assistant.
Use the provided CONTEXT to answer the user's question.
- If the context is insufficient, say what is missing and ask a follow-up question.
- Be concise and accurate.
- When you use context, cite sources like: [source: <filename>#<chunk_index>]
"""

def format_context(matches: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    ctx_blocks = []
    citations = []
    for m in matches:
        md = m.get("metadata") or {}
        source = md.get("source", "unknown")
        chunk_index = md.get("chunk_index", -1)
        text = (md.get("text") or "").strip()
        if not text:
            continue
        cite = f"{source}#{chunk_index}"
        citations.append(cite)
        ctx_blocks.append(f"[{cite}]\n{text}")
    return "\n\n".join(ctx_blocks), citations

class RAGAssistant:
    def __init__(self):
        self.embedder = SentenceTransformer(settings.EMBED_MODEL)
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX)

    def embed_query(self, q: str) -> List[float]:
        v = self.embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
        return v.astype(np.float32).tolist()

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        qv = self.embed_query(query)
        res = self.index.query(
            namespace=settings.PINECONE_NAMESPACE,
            vector=qv,
            top_k=top_k,
            include_metadata=True,
        )
        return res.get("matches", []) or []

    def ollama_chat(self, messages: List[Dict[str, str]]) -> str:
        url = settings.OLLAMA_BASE_URL.rstrip("/") + "/api/chat"
        payload = {
            "model": settings.OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content") or ""

    def answer(self, user_message: str, history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
        history = history or []
        matches = self.retrieve(user_message, settings.TOP_K)
        context, cites = format_context(matches[: settings.MAX_CONTEXT_CHUNKS])

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Keep a small history to avoid prompt bloat
        for m in history[-6:]:
            if "role" in m and "content" in m:
                messages.append({"role": m["role"], "content": m["content"]})

        if context:
            messages.append({"role": "system", "content": f"CONTEXT:\n{context}"})

        messages.append({"role": "user", "content": user_message})

        answer = self.ollama_chat(messages)

        return {
            "answer": answer,
            "sources": cites[: settings.MAX_CONTEXT_CHUNKS],
        }

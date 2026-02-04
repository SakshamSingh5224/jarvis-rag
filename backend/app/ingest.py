from __future__ import annotations

import hashlib
import time
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from .settings import settings

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    # Simple character-based chunking to keep dependencies minimal.
    # Works well enough for most docs; you can swap to token-based later.
    if not text:
        return []
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def make_id(source: str, chunk_index: int, chunk_text: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(str(chunk_index).encode("utf-8"))
    h.update(chunk_text.encode("utf-8"))
    return h.hexdigest()[:32]

class Ingestor:
    def __init__(self):
        self.embedder = SentenceTransformer(settings.EMBED_MODEL)
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX)

    def embed(self, texts: List[str]) -> List[List[float]]:
        embs = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embs.astype(np.float32).tolist()

    def upsert_document(self, *, text: str, source: str) -> Dict:
        chunks = chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        if not chunks:
            return {"ok": False, "message": "No text extracted", "chunks": 0}

        vectors = self.embed(chunks)

        items = []
        now = int(time.time())
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            _id = make_id(source, i, chunk)
            metadata = {
                "source": source,
                "chunk_index": i,
                "text": chunk,
                "ingested_at": now,
            }
            items.append((_id, vec, metadata))

        # Upsert in batches
        batch_size = 100
        for b in range(0, len(items), batch_size):
            self.index.upsert(
                vectors=items[b:b+batch_size],
                namespace=settings.PINECONE_NAMESPACE,
            )

        return {"ok": True, "message": "Upserted", "chunks": len(chunks), "source": source}

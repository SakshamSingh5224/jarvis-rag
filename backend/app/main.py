from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from .settings import settings
from .loaders import load_pdf_bytes, load_text_bytes
from .ingest import Ingestor
from .rag import RAGAssistant

app = FastAPI(title="Jarvis-RAG API", version="1.0.0")

ingestor = Ingestor()
assistant = RAGAssistant()

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    result = assistant.answer(req.message, req.history)
    return ChatResponse(**result)

@app.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)):
    data = await file.read()
    name = file.filename or "uploaded"
    lower = name.lower()

    if lower.endswith(".pdf"):
        doc = load_pdf_bytes(data, source=name)
    elif lower.endswith(".txt") or lower.endswith(".md"):
        doc = load_text_bytes(data, source=name)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, TXT, or MD.")

    if not doc.text:
        raise HTTPException(status_code=400, detail="No text extracted from file.")

    res = ingestor.upsert_document(text=doc.text, source=doc.source)
    if not res.get("ok"):
        raise HTTPException(status_code=500, detail=res.get("message", "Ingestion failed"))
    return res

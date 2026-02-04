from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from pypdf import PdfReader

@dataclass
class LoadedDoc:
    text: str
    source: str

def load_pdf_bytes(data: bytes, source: str) -> LoadedDoc:
    # pypdf can read from a file-like object; easiest is to write to temp is avoided.
    # PdfReader accepts a stream.
    import io
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n".join(pages).strip()
    return LoadedDoc(text=text, source=source)

def load_text_bytes(data: bytes, source: str) -> LoadedDoc:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="ignore")
    return LoadedDoc(text=text.strip(), source=source)

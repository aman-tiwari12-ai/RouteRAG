"""
ingestion.py — Document ingestion, chunking, embedding, and FAISS indexing.

Chunking strategy:
  - Chunk size: 512 tokens (~400 words). Large enough to preserve reasoning
    context (e.g., a full regulation clause + rationale), small enough that
    retrieved chunks don't dominate the context window.
  - Overlap: 80 tokens (~64 words). Prevents answers from being split across
    chunk boundaries—especially important for numbered lists and definitions.
  - Splitter: RecursiveCharacterTextSplitter with separators ["\n\n", "\n", " "]
    so paragraph breaks are preferred over mid-sentence cuts.
  - Metadata retained per chunk: source filename, chunk index, page estimate.
"""

import os
import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 512   # tokens approximated via char count (1 token ≈ 4 chars)
CHUNK_OVERLAP = 80
CHAR_PER_TOK  = 4
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast, OSS


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalise whitespace and strip common PDF artefacts."""
    text = re.sub(r'\x00', '', text)                   # null bytes
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)        # excess blank lines
    text = re.sub(r'[ \t]{2,}', ' ', text)             # double spaces
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]', '', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_txt(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def load_pdf(path: str) -> str:
    try:
        import PyPDF2
        text_parts = []
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_parts.append(page.extract_text() or '')
        return '\n\n'.join(text_parts)
    except Exception as e:
        print(f"[WARN] PDF load failed for {path}: {e}")
        return ""


def load_docx(path: str) -> str:
    try:
        from docx import Document
        doc = Document(path)
        return '\n\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        print(f"[WARN] DOCX load failed for {path}: {e}")
        return ""


def load_document(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == '.pdf':
        return clean_text(load_pdf(path))
    elif ext in ('.docx', '.doc'):
        return clean_text(load_docx(path))
    else:
        return clean_text(load_txt(path))


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_documents(docs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    docs: list of {"source": filename, "text": raw text}
    Returns list of chunk dicts with metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * CHAR_PER_TOK,
        chunk_overlap=CHUNK_OVERLAP * CHAR_PER_TOK,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc['text'])
        for i, split_text in enumerate(splits):
            chunk_id = hashlib.md5(f"{doc['source']}:{i}:{split_text[:50]}".encode()).hexdigest()[:12]
            chunks.append({
                "id":        chunk_id,
                "source":    doc['source'],
                "chunk_idx": i,
                "text":      split_text,
                "char_len":  len(split_text),
            })
    print(f"[Ingestion] {len(docs)} documents → {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# Embeddings + FAISS index
# ---------------------------------------------------------------------------
class VectorStore:
    def __init__(self, model_name: str = EMBED_MODEL):
        print(f"[VectorStore] Loading embedding model: {model_name}")
        self.embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.index   = None
        self.chunks  = []   # parallel list to FAISS rows
        self.dim     = 384  # all-MiniLM-L6-v2 output dim

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        texts = [c['text'] for c in chunks]
        print(f"[VectorStore] Embedding {len(texts)} chunks …")
        embeddings = self.embedder.embed_documents(texts)
        matrix = np.array(embeddings, dtype='float32')

        self.index  = faiss.IndexFlatIP(self.dim)   # inner product = cosine (normalised)
        self.index.add(matrix)
        self.chunks = chunks
        print(f"[VectorStore] FAISS index built — {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q_vec = np.array(
            self.embedder.embed_query(query), dtype='float32'
        ).reshape(1, -1)
        scores, indices = self.index.search(q_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk['similarity'] = float(score)
            results.append(chunk)
        return results

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.json"), 'w') as f:
            json.dump(self.chunks, f, indent=2)
        print(f"[VectorStore] Saved to {directory}/")

    def load(self, directory: str) -> None:
        self.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.json")) as f:
            self.chunks = json.load(f)
        print(f"[VectorStore] Loaded {self.index.ntotal} vectors from {directory}/")


# ---------------------------------------------------------------------------
# Entry point: ingest a folder of documents
# ---------------------------------------------------------------------------
def ingest_folder(data_dir: str, index_dir: str = "index") -> VectorStore:
    data_path = Path(data_dir)
    supported = {'.txt', '.pdf', '.docx', '.doc', '.md'}
    files = [p for p in data_path.iterdir() if p.suffix.lower() in supported]
    if not files:
        raise FileNotFoundError(f"No supported documents found in {data_dir}")

    docs = []
    for fp in files:
        print(f"[Ingestion] Loading {fp.name} …")
        text = load_document(str(fp))
        if text:
            docs.append({"source": fp.name, "text": text})
        else:
            print(f"[WARN] Empty document: {fp.name}")

    chunks = chunk_documents(docs)
    vs = VectorStore()
    vs.build(chunks)
    vs.save(index_dir)
    return vs


if __name__ == "__main__":
    import sys
    data_dir  = sys.argv[1] if len(sys.argv) > 1 else "data"
    index_dir = sys.argv[2] if len(sys.argv) > 2 else "index"
    ingest_folder(data_dir, index_dir)

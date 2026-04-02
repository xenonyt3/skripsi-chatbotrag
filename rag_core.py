"""
rag_core.py — RAG Engine dengan Ollama Embedding + ChromaDB
============================================================
Semantic search menggantikan TF-IDF untuk akurasi retrieval lebih baik.
"""
import os, re, json, requests
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import chromadb

# ---- Konfigurasi ----
CHROMA_DIR       = "chroma_db"
COLLECTION_NAME  = "rag_sop"
EMBED_MODEL      = "nomic-embed-text"
OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://localhost:11434")

@dataclass
class Chunk:
    doc_id: str
    judul: str
    kategori: str
    text: str

# ---- JSONL loader ----
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs

# ---- Chunking ----
def simple_chunk(text: str, max_chars: int = 400) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    chunks, buf = [], ""
    for s in sentences:
        if len(s) > max_chars:
            words = s.split()
            for w in words:
                cand = (buf + " " + w).strip() if buf else w
                if len(cand) <= max_chars:
                    buf = cand
                else:
                    if buf: chunks.append(buf.strip())
                    buf = w
        else:
            cand = (buf + " " + s).strip() if buf else s
            if len(cand) <= max_chars:
                buf = cand
            else:
                if buf: chunks.append(buf.strip())
                buf = s
    if buf: chunks.append(buf.strip())
    return [c for c in chunks if c]

# ---- Ollama Embedding ----
def _ollama_embed(texts: List[str], model: str = EMBED_MODEL, host: str = OLLAMA_HOST) -> List[List[float]]:
    """Kirim batch teks ke Ollama embedding API, return list of vectors."""
    try:
        r = requests.post(
            f"{host}/api/embed",
            json={"model": model, "input": texts},
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        if "embeddings" not in data:
            print(f"DEBUG: Ollama response missing 'embeddings': {data}")
            raise ValueError(f"Ollama response missing embeddings. Model {model} mungkin tidak support embedding.")
        return data["embeddings"]
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Gagal menghubungi Ollama ({host}): {e}")
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error in _ollama_embed: {e}")
        raise

# ---- ChromaDB client ----
def _get_chroma_collection(chroma_dir: str = CHROMA_DIR):
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

# ---- Build Index ----
def build_index(data_path: str, index_path: str = CHROMA_DIR, log_fn=None) -> None:
    """Build ChromaDB index dari file JSONL."""
    if log_fn is None:
        log_fn = print

    records = load_jsonl(data_path)
    chunks: List[Chunk] = []
    for r in records:
        for ch in simple_chunk(r["isi"]):
            chunks.append(Chunk(doc_id=r["id"], judul=r["judul"], kategori=r["kategori"], text=ch))

    if not chunks:
        raise ValueError("Tidak ada chunk yang dihasilkan dari data.")

    log_fn(f"   ↳ Total {len(chunks)} chunks, mulai embedding...")

    # Hapus collection lama jika ada
    client = chromadb.PersistentClient(path=index_path)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Batch embedding (maks 50 per batch agar tidak timeout)
    BATCH_SIZE = 50
    texts = [c.text for c in chunks]
    ids = [f"chunk-{i:05d}" for i in range(len(chunks))]
    metas = [{"doc_id": c.doc_id, "judul": c.judul, "kategori": c.kategori} for c in chunks]

    for start in range(0, len(texts), BATCH_SIZE):
        try:
            end = min(start + BATCH_SIZE, len(texts))
            batch_texts = texts[start:end]
            batch_ids = ids[start:end]
            batch_metas = metas[start:end]

            embeddings = _ollama_embed(batch_texts)
            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metas,
            )
            log_fn(f"   ↳ Embedded {end}/{len(texts)} chunks")
            print(f"DEBUG: Embedded {end}/{len(texts)} chunks")
        except Exception as e:
            print(f"ERROR: Gagal pada batch {start}-{end}: {e}")
            raise

    log_fn(f"   ↳ Index tersimpan di {index_path}/")

# ---- Load Index ----
def load_index(index_path: str = CHROMA_DIR):
    """Load ChromaDB collection. Return collection object."""
    return _get_chroma_collection(index_path)

# ---- Retrieve ----
def retrieve(query: str, top_k: int = 8, index_path: str = CHROMA_DIR) -> List[Tuple[float, Dict[str, Any], str]]:
    """Semantic search: embed query → ChromaDB similarity search."""
    collection = _get_chroma_collection(index_path)

    # Embed query
    q_emb = _ollama_embed([query])

    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - (distance/2)
            distance = results["distances"][0][i]
            similarity = 1 - (distance / 2)
            meta = results["metadatas"][0][i]
            hits.append((similarity, meta, doc))

    return hits

# ---- Formatting & helpers ----
def format_citations(hits) -> str:
    return "\n".join([f"- [{m['judul']} • {m['kategori']} | skor={s:.3f}] {t}" for s, m, t in hits])

def answer_extractive(query: str, top_k: int) -> str:
    hits = retrieve(query, top_k=top_k)
    if not hits:
        return "Tidak ada konteks yang cocok."
    ctx = "\n".join([h[2] for h in hits])
    return f"{ctx}\n\n---\nRujukan:\n{format_citations(hits)}"

def most_common_title(hits):
    from collections import Counter
    c = Counter([m['judul'] for _, m, _ in hits])
    return c.most_common(1)[0][0]

def get_full_section_by_title(title, index_path: str = CHROMA_DIR) -> str:
    """Ambil semua chunk dengan judul tertentu dari ChromaDB."""
    collection = _get_chroma_collection(index_path)
    results = collection.get(
        where={"judul": title},
        include=["documents"]
    )
    if results and results["documents"]:
        return "\n".join(results["documents"])
    return ""

def list_request(q: str) -> bool:
    q = q.lower()
    return any(k in q for k in ["semua", "daftar", "sebutkan", "list", "lengkap"]) \
           and any(k in q for k in ["peraturan", "aturan", "ketentuan"])

def answer_ollama(query: str, top_k: int, model: str = "qwen2.5:3b", history: str = "") -> str:
    hits = retrieve(query, top_k=top_k)
    def _clean(s): return re.sub(r"\s+", " ", s).strip()

    # default: ambil 5 chunk agar fokus
    ctx_text = "\n\n".join([f"- {_clean(t)}" for _,_,t in hits[:5]])

    if list_request(query):
        title = most_common_title(hits)
        full = get_full_section_by_title(title)
        if full: ctx_text = full

    style_system = (
        "Kamu asisten lab yang ramah. ATURAN PENTING:\n"
        "1. Jawab HANYA berdasarkan konteks yang diberikan di bawah.\n"
        "2. DILARANG KERAS mengarang atau menambahkan informasi yang TIDAK ADA di konteks.\n"
        "3. Jika konteks tidak memuat jawaban, katakan: "
        "'Maaf, aku tidak menemukan jawabannya di modul yang tersedia.'\n"
        "4. Jangan mencampuradukkan informasi dari topik berbeda.\n"
        "5. Abaikan riwayat percakapan saat menjawab, fokus hanya pada PERTANYAAN terbaru."
    )

    # Bangun prompt: riwayat opsional + konteks + pertanyaan
    history_block = f"[Riwayat percakapan]\n{history}\n\n" if history else ""
    prompt = (
        f"{history_block}"
        "Berikut konteks SOP.\n\n"
        f"{ctx_text}\n\n"
        f"PERTANYAAN: {query}\n\n"
        "Jika pertanyaan meminta 'semua/daftar' aturan, tampilkan SEMUA butir yang ada "
        "di konteks tanpa membuang poin. Bila tidak, ringkas seperlunya."
        "\nJAWAB (hanya berdasarkan konteks di atas):"
    )

    hosts = [os.getenv("OLLAMA_HOST","http://localhost:11434")]
    options = {"temperature": 0.1, "top_p": 0.9, "num_predict": 1024}

    last_err=None
    for host in hosts:
        try:
            requests.get(f"{host}/api/tags", timeout=2)
            r = requests.post(
                f"{host}/api/generate",
                json={"model": model, "prompt": prompt, "system": style_system,
                      "stream": False, "options": options},
                timeout=120,
            )
            r.raise_for_status()
            out = r.json().get("response","{} ").strip()
            return f"{out}\n\n---\nRujukan:\n{format_citations(hits)}"
        except requests.exceptions.RequestException as e:
            last_err=e; continue
    return f"❌ Tidak bisa terhubung ke Ollama. Detail: {last_err}"

# ---- CLI ----
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    sb = sub.add_parser("build-index", help="Bangun indeks embedding dari JSONL")
    sb.add_argument("--data", required=True, help="Path file JSONL, mis. data_all.jsonl")

    sa = sub.add_parser("ask", help="Tanya (extractive / ollama)")
    sa.add_argument("query")
    sa.add_argument("--mode", choices=["extractive", "ollama"], default="ollama")
    sa.add_argument("--top-k", type=int, default=8)
    sa.add_argument("--model", default="qwen2.5:3b")

    si = sub.add_parser("index-info", help="Info ringkas isi index")

    args = ap.parse_args()

    if args.cmd == "build-index":
        build_index(args.data)
        print(f"[OK] Index dibangun dari {args.data} -> {CHROMA_DIR}/")

    elif args.cmd == "ask":
        if args.mode == "extractive":
            print(answer_extractive(args.query, args.top_k))
        else:
            print(answer_ollama(args.query, args.top_k, model=args.model))

    elif args.cmd == "index-info":
        col = _get_chroma_collection()
        print(f"Index : {CHROMA_DIR}/ | chunks={col.count()}")
    else:
        ap.print_help()

import os, re, json, pickle, requests
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_FILE = "index.pkl"

@dataclass
class Chunk:
    doc_id: str
    judul: str
    kategori: str
    text: str

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs

def simple_chunk(text: str, max_chars: int = 400) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    chunks, buf = [], ""
    for s in sentences:
        cand = (buf + " " + s).strip() if buf else s
        if len(cand) <= max_chars:
            buf = cand
        else:
            if buf: chunks.append(buf.strip())
            buf = s
    if buf: chunks.append(buf.strip())
    return [c for c in chunks if c]

def build_index(data_path: str, index_path: str = INDEX_FILE) -> None:
    records = load_jsonl(data_path)
    chunks: List[Chunk] = []
    for r in records:
        for ch in simple_chunk(r["isi"]):
            chunks.append(Chunk(doc_id=r["id"], judul=r["judul"], kategori=r["kategori"], text=ch))

    texts = [c.text for c in chunks]
    meta  = [{"doc_id": c.doc_id, "judul": c.judul, "kategori": c.kategori} for c in chunks]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    X = vectorizer.fit_transform(texts)

    with open(index_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "X": X, "meta": meta, "texts": texts}, f)

def load_index(index_path: str = INDEX_FILE):
    with open(index_path, "rb") as f:
        obj = pickle.load(f)
    return obj["vectorizer"], obj["X"], obj["meta"], obj["texts"]

def retrieve(query: str, top_k: int = 8, index_path: str = INDEX_FILE) -> List[Tuple[float, Dict[str, Any], str]]:
    vectorizer, X, meta, texts = load_index(index_path)
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X)[0]
    idx = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), meta[i], texts[i]) for i in idx]

def format_citations(hits) -> str:
    return "\n".join([f"- [{m['judul']} • {m['kategori']} | skor={s:.3f}] {t}" for s, m, t in hits])

def answer_extractive(query: str, top_k: int) -> str:
    hits = retrieve(query, top_k=top_k)
    if not hits:
        return "Tidak ada konteks yang cocok."
    ctx = "\n".join([h[2] for h in hits])
    return f"{ctx}\n\n---\nRujukan:\n{format_citations(hits)}"

def most_common_title(hits):
    # hits: [(score, meta, text), ...]
    from collections import Counter
    c = Counter([m['judul'] for _, m, _ in hits])
    return c.most_common(1)[0][0]

def get_full_section_by_title(title, index_path: str = INDEX_FILE) -> str:
    vec, X, meta, texts = load_index(index_path)
    buf = []
    for i, m in enumerate(meta):
        if m['judul'].strip().lower() == title.strip().lower():
            buf.append(texts[i])
    return "\n".join(buf) if buf else ""


def list_request(q: str) -> bool:
    q = q.lower()
    return any(k in q for k in ["semua", "daftar", "sebutkan", "list", "lengkap"]) \
           and any(k in q for k in ["peraturan", "aturan", "ketentuan"])


def answer_ollama(query: str, top_k: int, model: str = "gpt-oss:20b") -> str:
    hits = retrieve(query, top_k=top_k)
    def _clean(s): return re.sub(r"\s+", " ", s).strip()

    # default: 3 potong saja biar fokus
    ctx_text = "\n\n".join([f"- {_clean(t)}" for _,_,t in hits[:3]])

    if list_request(query):
        title = most_common_title(hits)
        full = get_full_section_by_title(title)
        if full: ctx_text = full

    style_system = (
        "Kamu asisten lab yang ramah, jawab natural & to the point. "
        "Jangan mengarang di luar konteks. Jika tidak ada, ucapkan: "
        "'Maaf, aku tidak menemukan jawabannya di SOP yang tersedia.'"
    )

    prompt = (
        "Berikut konteks SOP.\n\n"
        f"{ctx_text}\n\n"
        f"PERTANYAAN: {query}\n\n"
        "Jika pertanyaan meminta 'semua/daftar' aturan, tampilkan SEMUA butir yang ada "
        "di konteks tanpa membuang poin. Bila tidak, ringkas seperlunya."
        "\nJAWAB:"
    )

    hosts = [os.getenv("OLLAMA_HOST","http://localhost:11434"), "http://0.0.0.0:11434"]
    options = {"temperature": 0.2, "top_p": 0.9, "num_predict": 512}

    last_err=None
    for host in hosts:
        try:
            requests.get(f"{host}/api/tags", timeout=2)
            r = requests.post(f"{host}/api/generate",
                json={"model": model, "prompt": prompt, "system": style_system,
                      "stream": False, "options": options},
                timeout=120)
            r.raise_for_status()
            out = r.json().get("response","").strip()
            return f"{out}\n\n---\nRujukan:\n{format_citations(hits)}"
        except requests.exceptions.RequestException as e:
            last_err=e; continue
    return f"❌ Tidak bisa terhubung ke Ollama. Detail: {last_err}"



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    sb = sub.add_parser("build-index", help="Bangun indeks TF-IDF dari JSONL")
    sb.add_argument("--data", required=True, help="Path file JSONL, mis. data_pti1.jsonl")

    sa = sub.add_parser("ask", help="Tanya (extractive / ollama)")
    sa.add_argument("query")
    sa.add_argument("--mode", choices=["extractive", "ollama"], default="ollama")
    sa.add_argument("--top-k", type=int, default=8)
    sa.add_argument("--model", default="gpt-oss:20b")

    si = sub.add_parser("index-info", help="Info ringkas isi index.pkl")

    args = ap.parse_args()

    if args.cmd == "build-index":
        build_index(args.data)
        print(f"[OK] Index dibangun dari {args.data} -> {INDEX_FILE}")

    elif args.cmd == "ask":
        if args.mode == "extractive":
            print(answer_extractive(args.query, args.top_k))
        else:
            print(answer_ollama(args.query, args.top_k, model=args.model))

    elif args.cmd == "index-info":
        
        with open(INDEX_FILE, "rb") as f:
            obj = pickle.load(f)
        num = len(obj.get("texts", []))
        print(f"Index : {INDEX_FILE} | chunks={num}")

    else:
        ap.print_help()


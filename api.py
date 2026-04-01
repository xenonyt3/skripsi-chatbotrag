import os
from collections import defaultdict, deque
from typing import Deque, Tuple, Dict

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_core import build_index, answer_extractive, answer_ollama

app = FastAPI(title="RAG Chat API", version="1.0")

# ---- CORS (atur domain produksi kalau sudah fix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Static UI (opsional)
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ---- Memory percakapan per-session
HIST: Dict[str, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=20))

# ---- Sapaan
GREETINGS = {
    "halo","hai","hi","hello","assalamualaikum","assalamu'alaikum",
    "pagi","siang","sore","malam","permisi"
}

def is_greeting(text: str) -> bool:
    t = (text or "").lower().strip()
    return t in GREETINGS or t.startswith("selamat ")

# ---- Models untuk request
class BuildReq(BaseModel):
    data_path: str = "data/data_all.jsonl"  # samakan dengan output GUI

class ChatReq(BaseModel):
    session_id: str
    message: str
    mode: str = "extractive"        # "extractive" | "ollama"
    top_k: int = 3
    model: str = "gpt-oss:20b"      # contoh default model

# ---- Health check
@app.get("/health")
def health():
    return {"ok": True}

# ---- Root: layani static/index.html jika ada
@app.get("/")
def root():
    if os.path.isfile("static/index.html"):
        return FileResponse("static/index.html")
    return {"ok": True, "msg": "RAG Chat API. Gunakan POST /build-index dan /chat"}

# ---- Build index
@app.post("/build-index")
def build(r: BuildReq):
    try:
        if not os.path.isfile(r.data_path):
            return JSONResponse({"ok": False, "error": f"File tak ditemukan: {r.data_path}"}, status_code=400)
        build_index(r.data_path)
        return {"ok": True, "msg": f"Index dibuat dari {r.data_path}"}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ---- Info index
@app.get("/index-info")
def index_info():
    try:
        import pickle
        obj = pickle.load(open("index.pkl", "rb"))
        return {
            "ok": True,
            "num_chunks": len(obj.get("texts", [])),
            "source_file": obj.get("source_file", "(tidak tersimpan)")
        }
    except FileNotFoundError:
        return JSONResponse({"ok": False, "error": "index.pkl belum ada. buat dulu via /build-index"}, status_code=404)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ---- Chat endpoint
@app.post("/chat")
def chat(r: ChatReq):
    try:
        sid = (r.session_id or "anon").strip()
        msg = (r.message or "").strip()
        if not msg:
            return {"answer": "tulis pesan dulu ya."}

        # reset sesi
        if msg.lower() in {"reset", "/reset"}:
            HIST[sid].clear()
            return {"answer": "✅ Sesi direset. Ada yang bisa saya bantu?"}

        # sapaan cepat
        if is_greeting(msg):
            reply = "Halo Praktikan PTI! 👋 Ada yang bisa saya bantu?"
            HIST[sid].append(("user", msg))
            HIST[sid].append(("assistant", reply))
            return {"answer": reply}

        # lanjut RAG
        HIST[sid].append(("user", msg))
        tail = "\n".join([f"{role.upper()}: {text}" for role, text in list(HIST[sid])[-6:]])

        if r.mode == "ollama":
            query = f"[Riwayat singkat]\n{tail}\n\n[Pesan terbaru]\n{msg}"
            ans = answer_ollama(query, r.top_k, model=r.model)
        else:
            ans = answer_extractive(msg, r.top_k)

        HIST[sid].append(("assistant", ans))
        return {"answer": ans}

    except Exception as e:
        return JSONResponse({"answer": f"Maaf, terjadi error: {e}"}, status_code=500)
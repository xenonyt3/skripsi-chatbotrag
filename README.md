# 🤖 Chatbot Asisten Lab Teknik — RAG + Ollama

Chatbot asisten praktikum berbasis **Retrieval-Augmented Generation (RAG)** dengan backend **FastAPI** dan LLM lokal via **Ollama**. Chatbot ini menjawab pertanyaan seputar SOP praktikum menggunakan data JSONL yang di-index dengan TF-IDF.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-orange)

---

## 📁 Struktur Proyek

```
CHATBOT ASISTEN LAB TEKNIK/
├── api.py                  # FastAPI server (endpoint /chat, /build-index, dll)
├── rag_core.py             # RAG engine (TF-IDF indexing + retrieval + Ollama)
├── requirements.txt        # Dependensi Python
├── run.bat                 # Script untuk jalankan Ollama + API server
├── run - ollama only.bat   # Script untuk jalankan Ollama saja
├── static/
│   └── index.html          # Web UI chatbot
├── data/                   # Folder data (opsional)
├── data_all.jsonl          # Dataset gabungan (SOP praktikum)
├── data_prokom.jsonl       # Dataset mata kuliah Prokom
├── BETAPROTO/              # Prototype versi beta (GUI desktop)
│   ├── app.py
│   ├── server.py
│   └── CONTROL PANEL.py
├── FINAL MODUL PROKOM 2025.json
├── FINAL MODUL PROKOM 2025.pdf
└── MODUL PTI 1 2024 DRAFT.pdf
```

> **Catatan:** Folder `.ollama/` dan `ollama/` (berisi model & binary Ollama) **tidak diupload ke GitHub** karena ukurannya besar. Kamu perlu install Ollama secara terpisah.

---

## 🚀 Setup di Laptop Baru

### Prasyarat

- **Python 3.10+** — [Download](https://www.python.org/downloads/)
- **Git** — [Download](https://git-scm.com/downloads)
- **Ollama** — [Download](https://ollama.com/download) (wajib untuk mode `ollama`)

---

### Langkah 1: Clone Repository

```bash
git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME
```

> Ganti `USERNAME/REPO_NAME` dengan URL repo GitHub-mu yang sebenarnya.

---

### Langkah 2: Install Python Dependencies

```bash
# Buat virtual environment (opsional tapi disarankan)
python -m venv .venv

# Aktifkan virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

# Install semua dependensi
pip install -r requirements.txt
```

---

### Langkah 3: Install & Setup Ollama

1. **Download & Install Ollama** dari [ollama.com](https://ollama.com/download)

2. **Pull model yang dibutuhkan:**
   ```bash
   ollama pull llama3.1
   ```
   > Atau model lain sesuai kebutuhanmu. Default di kode adalah `gpt-oss:20b`, sesuaikan di `api.py` bila perlu.

3. **Pastikan Ollama berjalan:**
   ```bash
   ollama serve
   ```
   Ollama akan listen di `http://localhost:11434`

---

### Langkah 4: Build Index

Index harus di-build ulang karena file `index.pkl` tidak diupload ke GitHub.

```bash
python rag_core.py build-index --data data_all.jsonl
```

Atau via API setelah server berjalan:
```bash
curl -X POST http://localhost:8000/build-index -H "Content-Type: application/json" -d "{\"data_path\": \"data_all.jsonl\"}"
```

---

### Langkah 5: Jalankan Server

**Cara Cepat (Windows):**
```bash
# Pastikan Ollama sudah running, lalu:
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Atau gunakan batch file** (otomatis start Ollama + API server):
```bash
run.bat
```

> ⚠️ Jika menggunakan `run.bat`, pastikan folder `ollama/` berisi binary `ollama.exe`. Jika tidak ada, install Ollama secara global dan jalankan Ollama + server terpisah.

---

### Langkah 6: Buka Chatbot

Buka browser dan akses:
```
http://localhost:8000
```

Web UI chatbot akan tampil. Pilih mode `ollama` atau `extractive`, lalu mulai bertanya!

---

## 🔧 Konfigurasi

### Mode Jawaban

| Mode | Deskripsi | Butuh Ollama? |
|------|-----------|:---:|
| `extractive` | Langsung menampilkan potongan teks relevan dari index | ❌ |
| `ollama` | Menggunakan LLM lokal (Ollama) untuk merangkum jawaban | ✅ |

### Environment Variable

| Variable | Default | Keterangan |
|----------|---------|------------|
| `OLLAMA_HOST` | `http://localhost:11434` | URL server Ollama |
| `OLLAMA_HOME` | (system default) | Path penyimpanan model Ollama |

---

## 📝 Format Data JSONL

Setiap baris di file `.jsonl` berbentuk:
```json
{"id": "sop-001", "judul": "Keselamatan Umum", "kategori": "keselamatan", "isi": "...paragraf SOP..."}
```

| Field | Keterangan |
|-------|------------|
| `id` | ID unik dokumen |
| `judul` | Judul/nama bagian SOP |
| `kategori` | Kategori SOP |
| `isi` | Isi lengkap teks SOP |

---

## 🛠️ API Endpoints

| Method | Endpoint | Keterangan |
|--------|----------|------------|
| `GET` | `/` | Web UI (static/index.html) |
| `GET` | `/health` | Health check |
| `POST` | `/build-index` | Build/rebuild index dari file JSONL |
| `GET` | `/index-info` | Info jumlah chunk di index |
| `POST` | `/chat` | Kirim pesan dan terima jawaban |

### Contoh Request `/chat`
```json
{
  "session_id": "user-123",
  "message": "Apa saja APD wajib di lab?",
  "mode": "ollama",
  "top_k": 3,
  "model": "llama3.1"
}
```

---

## ❓ Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `index.pkl belum ada` | Jalankan `python rag_core.py build-index --data data_all.jsonl` |
| `Tidak bisa terhubung ke Ollama` | Pastikan Ollama sudah jalan: `ollama serve` |
| `Model not found` | Pull model dulu: `ollama pull llama3.1` |
| `ModuleNotFoundError` | Jalankan `pip install -r requirements.txt` |

---

## 📄 Lisensi

Proyek ini untuk keperluan akademik — Praktikum Teknologi Informasi.

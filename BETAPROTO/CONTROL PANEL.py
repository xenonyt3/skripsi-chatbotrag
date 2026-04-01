# pdf2jsonl_gui.py (revisi)
import os, re, json, traceback, pickle
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from PyQt5 import QtWidgets, QtCore, QtGui

# ==== Try PyMuPDF (fitz) dulu; fallback ke PyPDF2 ====
try:
    import fitz  # PyMuPDF
    HAVE_FITZ = True
except Exception:
    HAVE_FITZ = False
    from PyPDF2 import PdfReader


try:
    from app import build_index as build_index_external
    HAVE_EXTERNAL_BUILD = True
except Exception:
    HAVE_EXTERNAL_BUILD = False

# ====== Implementasi internal build_index (backup) ======
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

INDEX_FILE_DEFAULT = "index.pkl"

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
            if line.strip():
                docs.append(json.loads(line))
    return docs

def simple_chunk_chars(text: str, max_chars: int = 450) -> List[str]:
    # fallback pemotongan per kalimat/paragraf bila tak terdeteksi daftar
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

def build_index_internal(data_path: str, index_path: str = INDEX_FILE_DEFAULT) -> None:
    records = load_jsonl(data_path)
    chunks: List[Chunk] = []
    for r in records:
        # r['isi'] sudah from GUI (hasil chunking sadar daftar)
        chunks.append(Chunk(doc_id=r["id"], judul=r["judul"], kategori=r["kategori"], text=r["isi"]))

    # ===== TF-IDF dengan judul di-boost (judul*2 + isi) =====
    texts = [((c.judul + " ") * 2 + c.text).strip() for c in chunks]
    meta  = [{"doc_id": c.doc_id, "judul": c.judul, "kategori": c.kategori} for c in chunks]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    X = vectorizer.fit_transform(texts)

    with open(index_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "X": X, "meta": meta, "texts": texts}, f)

def build_index(data_path: str, index_path: str = INDEX_FILE_DEFAULT) -> None:
    if HAVE_EXTERNAL_BUILD:
        return build_index_external(data_path, index_path)
    return build_index_internal(data_path, index_path)

# ====== Regex & helper kebersihan ======
SECTION_PATTERN = r"(^[A-Z][A-Z0-9 .\-]{3,}$|MODUL\s+\d+.*|^❖\s+[^\n]+|^•\s+[^\n]+|^▪\s+[^\n]+)"
RE_TOC_LINE     = re.compile(r'^[A-Za-z0-9].*\.{3,}\s+\d+\s*$')
RE_PAGE_NUM     = re.compile(r'^\s*\d+\s*$')
RE_DOT_LEADERS  = re.compile(r'\.{3,}')
RE_WS_MULTI     = re.compile(r'\s+')

def clean_line(s: str, drop_toc=True) -> str:
    s = s.strip()
    if not s:
        return ""
    if drop_toc and (RE_PAGE_NUM.match(s) or RE_TOC_LINE.match(s)):
        return ""
    s = RE_DOT_LEADERS.sub(' ', s)
    s = RE_WS_MULTI.sub(' ', s).strip()
    # buang header/footer umum
    if s.lower() in {"daftar isi", "table of contents"}:
        return ""
    return s

def read_pdf_text(path: str, drop_toc=True) -> str:
    """Ekstraksi teks per baris + kebersihan. PyMuPDF lebih stabil; fallback PyPDF2."""
    lines: List[str] = []
    if HAVE_FITZ:
        doc = fitz.open(path)
        for p in doc:
            text = p.get_text("text")
            for raw in text.splitlines():
                s = clean_line(raw, drop_toc=drop_toc)
                if s:
                    lines.append(s)
    else:
        reader = PdfReader(path)
        for p in reader.pages:
            text = p.extract_text() or ""
            for raw in text.splitlines():
                s = clean_line(raw, drop_toc=drop_toc)
                if s:
                    lines.append(s)
    return "\n".join(lines)

def split_sections(fulltext: str) -> List[Tuple[str, str]]:
    parts = re.split(SECTION_PATTERN, fulltext, flags=re.IGNORECASE)
    sections = []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i+1].strip() if i+1 < len(parts) else ""
        # Rapikan judul yang ketempelan nomor halaman
        title = RE_DOT_LEADERS.sub(' ', title)
        title = RE_WS_MULTI.sub(' ', title).strip()
        sections.append((title, content))
    if not sections:
        sections.append(("ISI DOKUMEN", fulltext))
    return sections

# === Chunking sadar-daftar ===
RE_NUM_ITEM   = re.compile(r"^\s*\d+[\.\)]\s+")
RE_ALPHA_ITEM = re.compile(r"^\s*[a-zA-Z][\.\)]\s+")
RE_BULLET     = re.compile(r"^\s*[•\-▪❖]\s+")

def chunk_list_aware(text: str, max_chars: int = 450) -> List[str]:
    """Pisah teks menjadi chunk dengan menjaga butir-butir list agar tidak terpecah."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    items, buf = [], []

    def flush():
        if buf:
            items.append(" ".join(buf).strip())

    def is_new_item(ln: str) -> bool:
        return bool(RE_NUM_ITEM.match(ln) or RE_ALPHA_ITEM.match(ln) or RE_BULLET.match(ln))

    for ln in lines:
        if is_new_item(ln):
            if buf: flush()
            buf = [ln]
        else:
            buf.append(ln)
    flush()

    if len(items) <= 1:
        # fallback: chunk biasa
        return simple_chunk_chars(text, max_chars=max_chars)

    chunks, cur = [], ""
    for it in items:
        if cur and len(cur) + 1 + len(it) <= max_chars:
            cur += " " + it
        else:
            if cur: chunks.append(cur)
            cur = it
    if cur: chunks.append(cur)
    return chunks


def guess_category(title: str) -> str:
    t = (title or "").lower()
    if "peraturan" in t: return "kedisiplinan"
    if "modul 1" in t or "desain" in t: return "desain"
    if "modul 2" in t or "bill of material" in t: return "bom"
    if "modul 3" in t or "lembar rencana proses" in t: return "lrp"
    if "modul 4" in t or "operation process chart" in t: return "opc"
    if "modul 5" in t or "assembly process chart" in t: return "apc"
    if "modul 6" in t or "computer numerical control" in t: return "cnc"
    if "referensi" in t: return "referensi"
    if "lampiran" in t: return "lampiran"
    return "umum"

# ====== Worker threads ======
class ConvertWorker(QtCore.QThread):
    log = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, pdf_paths: List[str], out_jsonl: str, max_chars: int = 450, clean_toc: bool = True):
        super().__init__()
        self.pdf_paths = pdf_paths
        self.out_jsonl = out_jsonl
        self.max_chars = max_chars
        self.clean_toc = clean_toc

    def run(self):
        try:
            total = 0
            with open(self.out_jsonl, "w", encoding="utf-8") as f:
                for path in self.pdf_paths:
                    base = os.path.basename(path)
                    self.log.emit(f"📄 Membaca: {base} (engine: {'PyMuPDF' if HAVE_FITZ else 'PyPDF2'})")
                    fulltext = read_pdf_text(path, drop_toc=self.clean_toc)
                    sections = split_sections(fulltext)
                    self.log.emit(f"   ↳ Deteksi {len(sections)} seksi")
                    i = 1
                    for title, body in sections:
                        chunks = chunk_list_aware(body, max_chars=self.max_chars)
                        for ch in chunks:
                            row = {
                                "id": f"{os.path.splitext(base)[0]}-{i:05d}",
                                "judul": title,
                                "kategori": guess_category(title),
                                "isi": ch
                            }
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            i += 1
                            total += 1
                    self.log.emit(f"   ↳ Selesai {base}: {i-1} chunk")
            self.done.emit(f"[OK] {total} chunk tersimpan ke {self.out_jsonl}")
        except Exception as e:
            self.error.emit(f"❌ Convert gagal: {e}\n{traceback.format_exc()}")

class BuildWorker(QtCore.QThread):
    log = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, jsonl_path: str, index_path: str):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.index_path = index_path

    def run(self):
        try:
            self.log.emit(f"🔧 Build index dari {self.jsonl_path} → {self.index_path}")
            build_index(self.jsonl_path, self.index_path)
            self.done.emit(f"[OK] Index siap: {self.index_path}")
        except Exception as e:
            self.error.emit(f"❌ Build index gagal: {e}\n{traceback.format_exc()}")

# ====== UI utama ======
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF → JSONL → Index (RAG SOP)")
        self.resize(860, 660)

        # Widgets
        self.pdf_list = QtWidgets.QListWidget()
        self.btn_add = QtWidgets.QPushButton("Tambah PDF…")
        self.btn_del = QtWidgets.QPushButton("Hapus")
        self.btn_clear = QtWidgets.QPushButton("Bersihkan")
        self.maxchars = QtWidgets.QSpinBox()
        self.maxchars.setRange(120, 4000)
        self.maxchars.setValue(450)
        self.out_jsonl = QtWidgets.QLineEdit("data_all.jsonl")
        self.btn_pick_out = QtWidgets.QPushButton("Simpan sebagai…")

        self.chk_autobuild = QtWidgets.QCheckBox("Auto build index setelah convert")
        self.chk_clean_toc = QtWidgets.QCheckBox("Bersihkan TOC/nomor halaman/header/footer")
        self.chk_clean_toc.setChecked(True)

        self.index_path = QtWidgets.QLineEdit(INDEX_FILE_DEFAULT)
        self.btn_pick_index = QtWidgets.QPushButton("Pilih index.pkl…")
        self.btn_convert = QtWidgets.QPushButton("Convert ke JSONL")
        self.btn_build = QtWidgets.QPushButton("Build Index sekarang")
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.hide()

        # Layout
        grid = QtWidgets.QGridLayout(self)
        row = 0
        grid.addWidget(QtWidgets.QLabel("Daftar PDF:"), row, 0)
        grid.addWidget(self.pdf_list, row, 1, 5, 3)
        vbtn = QtWidgets.QVBoxLayout()
        vbtn.addWidget(self.btn_add)
        vbtn.addWidget(self.btn_del)
        vbtn.addWidget(self.btn_clear)
        vbtn.addStretch()
        grid.addLayout(vbtn, row, 4, 5, 1)

        row += 5
        grid.addWidget(QtWidgets.QLabel("Max chars per chunk:"), row, 0)
        grid.addWidget(self.maxchars, row, 1)
        grid.addWidget(QtWidgets.QLabel("Output JSONL:"), row, 2)
        grid.addWidget(self.out_jsonl, row, 3)
        grid.addWidget(self.btn_pick_out, row, 4)

        row += 1
        grid.addWidget(self.chk_autobuild, row, 0, 1, 2)
        grid.addWidget(self.chk_clean_toc, row, 2, 1, 3)

        row += 1
        grid.addWidget(QtWidgets.QLabel("Index path:"), row, 0)
        grid.addWidget(self.index_path, row, 1, 1, 3)
        grid.addWidget(self.btn_pick_index, row, 4)

        row += 1
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.btn_convert)
        h.addWidget(self.btn_build)
        grid.addLayout(h, row, 0, 1, 5)

        row += 1
        grid.addWidget(QtWidgets.QLabel("Log:"), row, 0)
        row += 1
        grid.addWidget(self.log, row, 0, 1, 5)
        row += 1
        grid.addWidget(self.progress, row, 0, 1, 5)

        # Signals
        self.btn_add.clicked.connect(self.on_add)
        self.btn_del.clicked.connect(self.on_del)
        self.btn_clear.clicked.connect(self.pdf_list.clear)
        self.btn_pick_out.clicked.connect(self.on_pick_out)
        self.btn_pick_index.clicked.connect(self.on_pick_index)
        self.btn_convert.clicked.connect(self.on_convert)
        self.btn_build.clicked.connect(self.on_build)

    # ---------- UI handlers ----------
    def on_add(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Pilih PDF", "", "PDF Files (*.pdf)")
        for f in files:
            if f and not self._in_list(f):
                self.pdf_list.addItem(f)

    def on_del(self):
        for it in self.pdf_list.selectedItems():
            self.pdf_list.takeItem(self.pdf_list.row(it))

    def on_pick_out(self):
        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Simpan JSONL", self.out_jsonl.text(), "JSONL (*.jsonl)")
        if f:
            if not f.lower().endswith(".jsonl"):
                f += ".jsonl"
            self.out_jsonl.setText(f)

    def on_pick_index(self):
        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Simpan index.pkl", self.index_path.text(), "Pickle (*.pkl)")
        if f:
            if not f.lower().endswith(".pkl"):
                f += ".pkl"
            self.index_path.setText(f)

    def on_convert(self):
        pdfs = [self.pdf_list.item(i).text() for i in range(self.pdf_list.count())]
        if not pdfs:
            self._append("❗ Tambahkan PDF dulu.")
            return
        out = self.out_jsonl.text().strip()
        if not out:
            self._append("❗ Tentukan output JSONL.")
            return
        maxchars = int(self.maxchars.value())
        self._run_convert(pdfs, out, maxchars, self.chk_clean_toc.isChecked())

    def on_build(self):
        jsonl = self.out_jsonl.text().strip()
        if not jsonl or not os.path.isfile(jsonl):
            jsonl, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Pilih JSONL untuk build index", "", "JSONL (*.jsonl)")
            if not jsonl:
                self._append("❗ Pilih file JSONL dulu.")
                return
            self.out_jsonl.setText(jsonl)
        index_path = self.index_path.text().strip() or INDEX_FILE_DEFAULT
        self._run_build(jsonl, index_path)

    # ---------- worker runners ----------
    def _run_convert(self, pdfs: List[str], out: str, maxchars: int, clean_toc: bool):
        self.progress.show()
        self._append(f"▶️ Convert {len(pdfs)} PDF → {out} (max {maxchars} chars/chunk; clean_toc={clean_toc})")
        self.btn_convert.setEnabled(False)
        self.btn_build.setEnabled(False)
        self.worker_c = ConvertWorker(pdfs, out, max_chars=maxchars, clean_toc=clean_toc)
        self.worker_c.log.connect(self._append)
        self.worker_c.done.connect(self._on_convert_done)
        self.worker_c.error.connect(self._on_error)
        self.worker_c.start()

    def _run_build(self, jsonl: str, index_path: str):
        self.progress.show()
        self._append(f"▶️ Build index dari {jsonl} → {index_path}")
        self.btn_convert.setEnabled(False)
        self.btn_build.setEnabled(False)
        self.worker_b = BuildWorker(jsonl, index_path)
        self.worker_b.log.connect(self._append)
        self.worker_b.done.connect(self._on_build_done)
        self.worker_b.error.connect(self._on_error)
        self.worker_b.start()

    # ---------- worker callbacks ----------
    def _on_convert_done(self, msg: str):
        self._append(msg)
        self.progress.hide()
        self.btn_convert.setEnabled(True)
        self.btn_build.setEnabled(True)
        if self.chk_autobuild.isChecked():
            self._run_build(self.out_jsonl.text().strip(), self.index_path.text().strip() or INDEX_FILE_DEFAULT)

    def _on_build_done(self, msg: str):
        self._append(msg)
        self.progress.hide()
        self.btn_convert.setEnabled(True)
        self.btn_build.setEnabled(True)

    def _on_error(self, msg: str):
        self._append(msg)
        self.progress.hide()
        self.btn_convert.setEnabled(True)
        self.btn_build.setEnabled(True)

    # ---------- utils ----------
    def _append(self, s: str):
        self.log.appendPlainText(s)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _in_list(self, path: str) -> bool:
        for i in range(self.pdf_list.count()):
            if self.pdf_list.item(i).text() == path:
                return True
        return False

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.setWindowIcon(QtGui.QIcon.fromTheme("document-save"))
    w.show()
    sys.exit(app.exec_())

# main.py: pusat aplikasi (service api). Menerima pertanyaan dari web chat,
# menjalankan RAG, lalu mengirim jawaban Llama secara streaming.

# ===== Import =====
import asyncio                                   # menjalankan fungsi sinkron tanpa memblokir server
import json                                      # membaca balasan Ollama yang berbentuk JSON
import os                                        # membaca environment variable
from typing import Literal                       # membatasi nilai yang diperbolehkan

import httpx                                     # mengirim request HTTP ke Ollama
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse  # respons yang dikirim bertahap
from fastapi.templating import Jinja2Templates   # menampilkan halaman HTML dari template
from pydantic import BaseModel                   # cetakan data masuk

from app.embeddings import embed_text            # teks -> 768 angka
from app.ingest import ingest_documents
from app.vector_store import VectorStore         # penghubung ke OpenSearch

# ===== Inisialisasi aplikasi =====
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")  # folder berisi chat.html

# ===== Konfigurasi dari environment (diisi di docker-compose.yml) =====
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")               # model chat
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")  # model embedding
KNOWLEDGE_BASE_PATH = os.environ.get("KNOWLEDGE_BASE_PATH", "/app/knowledge-base")
OPENSEARCH_BASE_URL = os.environ.get("OPENSEARCH_BASE_URL", "http://localhost:9200")
OPENSEARCH_INDEX = os.environ.get("OPENSEARCH_INDEX", "bumbuai-docs")

# ===== Pengaturan RAG =====
RAG_TOP_K = 3                  # ambil 3 resep teratas dari pencarian
RAG_SCORE_THRESHOLD = 0.72     # resep dengan skor di bawah ini dibuang
HISTORY_WINDOW = 10            # hanya 10 pesan terakhir yang dikirim ke model
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1500"))  # batas panjang token jawaban model

# Objek penghubung ke OpenSearch (dibuat sekali, dipakai tiap request).
vector_store = VectorStore(base_url=OPENSEARCH_BASE_URL)

# ===== Aturan perilaku BumbuAI =====
SYSTEM_PROMPT = (
    "Kamu adalah BumbuAI, asisten AI resep masakan Indonesia yang dibuat oleh Tim BumbuAI.\n\n"
    "Pedoman Perilaku:\n"
    "- Identitas: Nama kamu adalah BumbuAI. Jika ditanya 'siapa kamu', 'siapa namamu', atau tentang pembuatmu, selalu jawab dengan ramah: "
    "'Aku BumbuAI, asisten resep makanan yang dibuat oleh Tim BumbuAI! Ada resep yang ingin kamu tanyakan hari ini?'\n"
    "- Batasan Topik: Hanya jawab seputar resep makanan, bahan, dan cara memasak. "
    "Jika pertanyaan di luar itu (politik, sejarah, tokoh, berita, dan sebagainya), JANGAN menjawab isinya sama sekali. "
    "Jawab hanya dengan: 'Maaf, aku hanya bisa membantu soal resep makanan. Mau masak apa hari ini?'\n"
)

# Tombol saran pertanyaan di halaman awal chat.
SUGGESTIONS = [
    "Bagaimana cara membuat nasi goreng kampung?",
    "Resep sambal terasi yang simple",
    "Ide menu sarapan dari telur dan tempe",
]


# ===== Cetakan data masuk =====
class Message(BaseModel):
    role: Literal["user", "assistant"]   # siapa yang bicara
    content: str                         # isi pesannya


class ChatRequest(BaseModel):
    messages: list[Message]              # seluruh riwayat percakapan


class EmbedRequest(BaseModel):
    text: str                            # teks yang mau di-embed (untuk /embed)


# ===== GET /: menampilkan halaman web chat =====
@app.get("/", include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "chat.html",
        {"app_name": "BumbuAI", "model": OLLAMA_MODEL, "suggestions": SUGGESTIONS},
    )


# ===== GET /health: cek server hidup =====
@app.get("/health")
def health_check():
    return {"status": "ok"}


# ===== POST /chat: inti RAG + streaming =====
@app.post("/chat")
async def chat(req: ChatRequest):

    # 1) Validasi: tolak jika kosong atau pesan terakhir bukan dari user.
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail="messages harus non-kosong dan elemen terakhir harus role 'user'",
        )

    # 2) Ambil 10 pesan terakhir supaya prompt tidak terlalu panjang.
    recent = req.messages[-HISTORY_WINDOW:]

    # 3) RAG: ubah pertanyaan jadi vektor, lalu cari resep terdekat.
    query_embedding = await asyncio.to_thread(
        embed_text, recent[-1].content, OLLAMA_BASE_URL
    )
    hits = await asyncio.to_thread(
        vector_store.search, OPENSEARCH_INDEX, query_embedding, RAG_TOP_K
    )

    # 4) Saring: hanya resep dengan skor cukup tinggi yang dipakai.
    system_prompt = SYSTEM_PROMPT
    relevant_hits = [h for h in hits if h.get("_score", 0) >= RAG_SCORE_THRESHOLD]

    # 5) Tempel resep yang lolos ke prompt sebagai konteks.
    if relevant_hits:
        context = "\n\n".join(f"- {h['_source']['text']}" for h in relevant_hits)
        system_prompt += f"\n\nKonteks dari knowledge base:\n{context}"

    # 6) Pengingat di akhir prompt (model kecil mudah lupa aturan di awal).
    system_prompt += "\n\nIngat: kamu BumbuAI. Jika pertanyaan bukan tentang resep makanan, jawab hanya dengan kalimat penolakan di atas."

    # 7) Susun pesan untuk Ollama: system prompt + riwayat chat.
    messages = [{"role": "system", "content": system_prompt}]
    messages += [m.model_dump() for m in recent]
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,   # minta Ollama membalas sepotong-sepotong
        "options": {"temperature": 0.3, "num_ctx": 4096, "num_predict": OLLAMA_NUM_PREDICT},
    }

    # 8) Kirim ke Ollama, lalu teruskan jawabannya potongan demi potongan.
    async def stream():
        try:
            # Buka koneksi ke Ollama tanpa batas waktu (jawaban bisa lama).
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST", f"{OLLAMA_BASE_URL}/api/chat", json=payload
                ) as r:
                    # Kalau Ollama menolak (misalnya model belum di-pull), kirim pesan error.
                    if r.status_code != 200:
                        yield f"[Error {r.status_code}] Model belum tersedia di Ollama."
                        return
                    # Baca balasan baris demi baris, setiap baris satu potongan jawaban.
                    async for line in r.aiter_lines():
                        if not line:
                            continue   # lewati baris kosong
                        # Ambil teks dari JSON, lalu kirim ke browser saat itu juga.
                        piece = json.loads(line).get("message", {}).get("content", "")
                        if piece:
                            yield piece
        except httpx.HTTPError as e:
            # Ollama mati atau tidak terjangkau.
            yield f"[Error] Tidak bisa terhubung ke Ollama: {e}"

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


# ===== POST /embed: endpoint uji embedding  =====
@app.post("/embed")
async def embed(req: EmbedRequest):
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": req.text},
        )
        r.raise_for_status()
        vector = r.json()["embeddings"][0]
    return {"model": OLLAMA_EMBED_MODEL, "dimensions": len(vector), "embedding": vector}


# ===== POST /ingest: memproses semua file resep di knowledge-base =====
@app.post("/ingest")
async def ingest():
    count = await asyncio.to_thread(
        ingest_documents,
        KNOWLEDGE_BASE_PATH,
        vector_store,
        OPENSEARCH_INDEX,
        OLLAMA_BASE_URL,
    )
    return {
        "status": "ok",
        "message": f"Berhasil mengindeks {count} file resep dari knowledge-base ke OpenSearch.",
        "indexed_documents": count,
    }

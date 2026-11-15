# ingest.py: memasukkan file resep di knowledge-base/ ke OpenSearch.
import os

from pypdf import PdfReader

from app.embeddings import embed_text
from app.vector_store import VectorStore


# extract_text: membaca isi sebuah file menjadi teks.
def extract_text(file_path: str) -> str:
    # Kalau file PDF, baca halaman demi halaman lalu gabungkan.
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    # Selain PDF (misalnya .md), baca isinya apa adanya.
    with open(file_path, "r") as f:
        return f.read()


# ingest_documents: memproses semua file di folder, mengembalikan jumlah file.
def ingest_documents(
    knowledge_base_path: str,
    vector_store: VectorStore,
    index_name: str,
    ollama_base_url: str,
) -> int:
    # Pastikan index di OpenSearch sudah ada (kalau belum, dibuat).
    vector_store.ensure_index(index_name)
    count = 0

    # Ulangi untuk setiap file di folder, urut sesuai abjad.
    for filename in sorted(os.listdir(knowledge_base_path)):
        file_path = os.path.join(knowledge_base_path, filename)

        # Lewati kalau bukan file (misalnya subfolder).
        if not os.path.isfile(file_path):
            continue

        # 1) Baca teks, 2) ubah jadi 768 angka, 3) simpan ke OpenSearch.
        text = extract_text(file_path)
        embedding = embed_text(text, base_url=ollama_base_url)
        vector_store.index_document(
            index_name,
            doc_id=filename,
            text=text,
            embedding=embedding,
            metadata={"source": filename},
        )
        count += 1

    return count


if __name__ == "__main__":
    kb_path = os.getenv("KNOWLEDGE_BASE_PATH", "knowledge-base")
    opensearch_url = os.getenv("OPENSEARCH_BASE_URL", "http://localhost:9200")
    index_name = os.getenv("OPENSEARCH_INDEX", "bumbuai-docs")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    print(f"Memulai ingest dari folder: {kb_path}...")
    store = VectorStore(base_url=opensearch_url)
    total = ingest_documents(kb_path, store, index_name, ollama_url)
    print(f"Selesai! Berhasil mengindeks {total} dokumen ke OpenSearch '{index_name}'.")

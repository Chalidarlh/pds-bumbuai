# embeddings.py: berisi fungsi untuk mengubah teks menjadi embedding
# (768 angka yang mewakili makna teks).
import httpx

# embed_text: mengirim teks ke Ollama, mengembalikan embedding-nya.
def embed_text(text: str, base_url: str, model: str = "nomic-embed-text") -> list[float]:

    # Buka koneksi HTTP ke Ollama (otomatis ditutup setelah selesai).
    with httpx.Client() as client:

        # Kirim teks ke Ollama, tunggu maksimal 60 detik.
        response = client.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": text},
            timeout=60.0,
        )

        # Kalau Ollama membalas error, langsung hentikan dengan pesan jelas.
        response.raise_for_status()

        # Ambil hasilnya: satu daftar berisi 768 angka.
        return response.json()["embeddings"][0]
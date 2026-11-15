# vector_store.py: perantara antara kode kita dan OpenSearch.
import httpx


# Class = "paket" berisi fungsi-fungsi yang berhubungan dengan OpenSearch.
class VectorStore:

    # Dijalankan saat objek dibuat: simpan alamat OpenSearch dan jumlah dimensi vektor.
    def __init__(self, base_url: str, dimension: int = 768):
        self.base_url = base_url
        self.dimension = dimension

    # ensure_index: pastikan "tabel" (index) untuk resep sudah ada.
    def ensure_index(self, index_name: str) -> None:
        with httpx.Client() as client:
            # Tanya OpenSearch: index ini sudah ada? (200 = sudah ada)
            existing = client.head(f"{self.base_url}/{index_name}", timeout=30.0)
            if existing.status_code == 200:
                return  # sudah ada, tidak perlu apa-apa

            # Belum ada, jadi buat index baru beserta aturan penyimpanannya.
            response = client.put(
                f"{self.base_url}/{index_name}",
                json={
                    # Aktifkan fitur pencarian vektor (k-NN).
                    "settings": {"index": {"knn": True}},
                    "mappings": {
                        "properties": {
                            # Kolom teks resep.
                            "text": {"type": "text"},
                            # Kolom vektor: 768 angka, dicari pakai cosine similarity.
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": self.dimension,
                                "method": {
                                    "engine": "nmslib",
                                    "space_type": "cosinesimil",
                                    "name": "hnsw",
                                },
                            },
                            # Kolom data tambahan (misalnya nama file sumber).
                            "metadata": {"type": "object"},
                        }
                    },
                },
                timeout=30.0,
            )
            response.raise_for_status()

    # index_document: simpan satu dokumen (teks + vektor + metadata).
    def index_document(
        self,
        index_name: str,
        doc_id: str,
        text: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        with httpx.Client() as client:
            # PUT ke alamat dengan doc_id: kalau ID sudah ada, dokumen lama tertimpa.
            response = client.put(
                f"{self.base_url}/{index_name}/_doc/{doc_id}",
                params={"refresh": "true"},  # langsung bisa dicari, tanpa jeda
                json={"text": text, "embedding": embedding, "metadata": metadata},
                timeout=30.0,
            )
            response.raise_for_status()

    # search: cari k dokumen yang vektornya paling dekat dengan vektor pertanyaan.
    def search(
        self, index_name: str, query_embedding: list[float], k: int = 3
    ) -> list[dict]:
        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/{index_name}/_search",
                json={
                    "size": k,  # jumlah hasil yang dikembalikan
                    "query": {"knn": {"embedding": {"vector": query_embedding, "k": k}}},
                },
                timeout=30.0,
            )
            response.raise_for_status()
            # Kembalikan daftar hasil mentah (berisi _id, _score, _source).
            return response.json()["hits"]["hits"]
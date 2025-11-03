import os
import time
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import ollama

# -----------------------------
# 1. AYARLAR
# -----------------------------
JSONL_FILE = "mevzuat_rag_data.jsonl"
TOP_K = 10
MAX_CONTEXT_LENGTH = 650
MAX_OUTPUT_TOKENS = 400

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
INDEX_FILE = "faiss_index_e5.bin"
EMBEDDINGS_FILE = "embeddings_e5.npy"

OLLAMA_MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"
OLLAMA_URL = "http://localhost:11434"

# -----------------------------
# 2. Cihaz seçimi (CPU) Veya cuda
# -----------------------------
device = "cpu"
print(f"Çalışma cihazı: {device}")
print(f"LLM, OLLAMA/{OLLAMA_MODEL_NAME} üzerinden {OLLAMA_URL} adresinde çalışacaktır.")

# -----------------------------
# 3. JSONL dosyalarını oku
# -----------------------------
texts, sources = [], []

if not os.path.exists(JSONL_FILE):
    print(f"\nHATA: '{JSONL_FILE}' bulunamadı!")
    exit()

try:
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            texts.append(doc["text"])
            meta = doc["metadata"]
            source_id = f"{meta['kanun_no']}_M{meta['madde_no']}_C{meta['chunk_id']}"
            sources.append(source_id)
    print(f"\nJSONL dosyasından {len(texts)} parça yüklendi.")
except Exception as e:
    print(f"HATA JSONL okuma: {e}")
    exit()

# -----------------------------
# 4. Embedding ve FAISS CPU
# -----------------------------
embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)

rebuild_index = True

if os.path.exists(INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
    try:
        embeddings = np.load(EMBEDDINGS_FILE)
        if embeddings.shape[0] == len(texts):
            index = faiss.read_index(INDEX_FILE)
            rebuild_index = False
            print("\nMevcut Vektör İndeksi Yükleniyor...")
        else:
            print("\n⚠️ Embedding sayısı JSONL ile uyuşmuyor. Yeniden oluşturulacak.")
    except:
        pass

if rebuild_index:
    print("\n✅ Vektör İndeksi Yeniden Oluşturuluyor...")
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    np.save(EMBEDDINGS_FILE, embeddings)
    faiss.write_index(index, INDEX_FILE)
    print("✅ İndeks ve Embeddings kaydedildi!")

# -----------------------------
# 5. Soru-cevap fonksiyonu
# -----------------------------
def answer_question(question, top_k=TOP_K, max_output_tokens=MAX_OUTPUT_TOKENS):
    q_emb = embedder.encode([question], convert_to_numpy=True).astype(np.float32)
    D, I = index.search(q_emb, top_k)
    indices = I[0].tolist()

    context_list = [f"{texts[i]} (Kaynak: {sources[i]})" for i in indices]
    context = "\n---\n".join(context_list)

    prompt = (
        "Sen bir Türk Vergi Mevzuatı uzmanısın. Yalnızca aşağıdaki KANUN BAĞLAMI içinde yer alan bilgilere dayanarak soruyu TÜRKÇE yanıtla.\n"
        "Cevabı maddeler halinde yaz ve her maddenin sonunda ilgili kaynağı parantez içinde belirt.\n"
        "Parantez içindeki (değişiklik, tarih, Kanun numarası) gibi metinleri CEZAYA dahil etme.\n\n"
        f"KANUN BAĞLAMI:\n{context}\n\nSORU: {question}"
    )

    t_api = time.time()
    try:
        client = ollama.Client(host=OLLAMA_URL)
        response = client.generate(
            model=OLLAMA_MODEL_NAME,
            prompt=prompt,
            stream=False,
            options={
                'num_predict': max_output_tokens,
                'temperature': 0.3,
                'repeat_penalty': 1.1,
            }
        )
        answer = response['response'].strip()
        print(f"API Cevap Süresi: {time.time() - t_api:.2f}s")
    except Exception as e:
        answer = f"HATA: Ollama API başarısız. Detay: {e}"

    used_sources = [sources[i] for i in indices]
    return answer, used_sources


# -----------------------------
# 6. İnteraktif döngü
# -----------------------------
if __name__ == "__main__":
    print("\n✅ Hazır — Soru sor (çıkmak için 'exit' yaz):")
    sample_q = "Katma Değer Vergisi Kanununa göre ihracat teslimleri nasıl istisnadır?"

    while True:
        q = input("\nSoru: ").strip()
        if not q:
            q = sample_q
        if q.lower() in ("exit", "quit", "çık", "q"):
            print("Çıkılıyor...")
            break

        t0 = time.time()
        ans, used = answer_question(q)

        print("\n📌 Cevap:\n", ans)
        print("\n📌 Kullanılan Kaynaklar:", used)
        print(f"(⏱ Toplam Süre: {time.time() - t0:.2f}s)")

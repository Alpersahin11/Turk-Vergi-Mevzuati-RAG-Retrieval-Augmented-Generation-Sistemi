# Türk Vergi Mevzuatı RAG Sistemi

Bu proje, Gelir İdaresi Başkanlığından indirilen mevzuat PDF'lerinden çıkarılan metinlerle çalışan bir RAG (Retrieval-Augmented Generation) tabanlı soru-cevap sistemidir.
Sistem, Türk vergi mevzuatına dayanarak kaynaklı yanıtlar üretir.

------------------------------------------------------------
🚀 Özellikler
------------------------------------------------------------
- Vergi mevzuatına göre doğru kaynaklı yanıt
- JSONL + FAISS ile hızlı vektör arama
- Multilingual embedding modeli (E5-Large)
- Ollama üzerinde Qwen 2.5 ile cevaplama
- CPU üzerinde çalışabilir
- PDF → JSONL → Embed → Index → Cevap

------------------------------------------------------------
📂 Proje Dosyaları
------------------------------------------------------------
- mevzuat_rag_data.jsonl → Mevzuat metinleri + metadata
- faiss_index_e5.bin → FAISS vektör index
- embeddings_e5.npy → Embedding dosyası
- rag_mevzuat.py → Ana çalışma dosyası
- README.txt → Bu açıklama dosyası

------------------------------------------------------------
🔧 Kurulum
------------------------------------------------------------
Gerekli kütüphaneleri kur:
pip install sentence-transformers faiss-cpu ollama numpy

Ollama modelini indir:
ollama pull qwen2.5:7b-instruct-q4_K_M

------------------------------------------------------------
▶️ Çalıştırma
------------------------------------------------------------
python rag_mevzuat.py

Soru sorarak başla:
Katma Değer Vergisi Kanununa göre ihracat teslimleri nasıl istisnadır?

Çıkmak için:
exit, quit, q, çık

------------------------------------------------------------
🎯 Örnek Cevap
------------------------------------------------------------
📌 Cevap:
- İhracat teslimleri KDV’den istisnadır. (Kaynak: 3065_M11_C2)
- Teslim Türkiye’de yapılsa bile istisna uygulanır. (Kaynak: 3065_M12_C1)

📌 Kullanılan Kaynaklar:
3065_M11_C2, 3065_M12_C1

------------------------------------------------------------
Mimari Özet
------------------------------------------------------------
Kullanıcı → Soru
↓
Embedding
↓
FAISS Arama (Top-K)
↓
Bulunan kanun maddeleri bağlam olarak modele verilir
↓
Kaynak gösteren LLM yanıtı

------------------------------------------------------------
Sorumluluk Reddi
------------------------------------------------------------
Bu proje hukuki danışmanlık amacı taşımaz.
Yanıtlar resmi görüş yerine geçmez.


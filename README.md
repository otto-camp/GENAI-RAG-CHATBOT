# 💳 FinAI

Finansal karar destek senaryolarına odaklanan, Google Gemini modelleriyle çalışan
retrieval-augmented generation (RAG) sohbet asistanı. Hugging Face üzerindeki finans
veri setleri vektörleştirilip hafif bir dosya tabanlı depo (`vector_store/`) içinde
tutuluyor; bağlam bulunamazsa model kontrollü şekilde genel bilgisini kullanıyor.

https://finai-gemini.streamlit.app/

## ✨ Öne Çıkanlar
- 💬 Streamlit tabanlı sohbet arayüzü (`app.py`) ve hafif tema dokunuşu (`styles/finai_theme.css`)
- 🧠 Hibrit arama: sıkı benzerlik eşiği + ihtiyaç duyulduğunda fallback taraması
- 🗂️ Dosya tabanlı vektör deposu (NumPy + JSONL) ve ilk açılışta otomatik ingest denemesi
- 🔁 Sohbet geçmişi, bağlam ve kurallarla dinamik prompt üretimi (`rag_utils.build_prompt`)
- 🔌 İki embedding modu: yerel `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  veya Gemini `text-embedding-004`
- 📚 Finans odaklı kaynaklar:
  - `Josephgflowers/Finance-Instruct-500k`
  - `gretelai/synthetic_pii_finance_multilingual`
  - (Opsiyonel) `CFPB/consumer-finance-complaints`

## 🗃️ Proje Yapısı
```text
FinAI/
├─ app.py                 # Streamlit sohbet arayüzü ve RAG akışı
├─ ingest.py              # Finans veri setlerini işleyip vector_store'a yazar
├─ rag_utils.py           # Gemini istemcisi, embedding yardımcıları ve prompt kuralları
├─ styles/
│  └─ finai_theme.css     # Hero alanı için hafif tema
├─ vector_store/          # ingest sonrası oluşur (embeddings.npy + documents.jsonl)
├─ requirements.txt       # Python bağımlılıkları
└─ .env.example           # Ortam değişkeni şablonu
```

> `vector_store/` klasörü repo içinde tutulmaz; ingest çalışınca oluşturulur.

## ⚙️ Kurulum
1. Depoyu klonlayın ve dizine girin.
2. (Önerilen) Sanal ortam kurun:
   ```bash
   python -m venv venv
   source venv/bin/activate     # Windows: venv\Scripts\activate
   ```
3. Bağımlılıkları yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

## 🔐 Ortam Değişkenleri
`.env.example` dosyasını `.env` olarak kopyalayın ve değerleri düzenleyin.

```bash
cp .env.example .env
```

- **Gemini**
  - `GOOGLE_API_KEY`: Zorunlu
  - `GOOGLE_LLM_MODEL`: Varsayılan `models/gemini-2.0-flash`
  - `GOOGLE_MAX_OUTPUT_TOKENS`: Maksimum token sayısı (varsayılan 512)
- **Embedding & veri**
  - `EMBED_PROVIDER`: `local` (varsayılan) veya `gemini`
  - `HF_LIMIT`: Her veri setinden alınacak maksimum kayıt (≤0 ise limitsiz)
  - `EMBED_BATCH_SIZE`: Yerel model için toplu iş boyutu
  - `EMBED_SLEEP_BETWEEN`: Gemini embed modunda istekler arası bekleme
  - `INCLUDE_CFPB_DATASET`: `true/false` (isteğe bağlı şikayet verisi)
  - `HF_DATASETS_OFFLINE`: `1` setlenirse Hugging Face cache’inden okur

## 📥 Veri Hazırlama (Ingest)
Hugging Face veri setlerini indirip vektör deposu oluşturmak için:

```bash
python ingest.py
```

Komut:
- Verileri normalize eder ve tekrar eden kayıtları temizler.
- Seçilen embedding sağlayıcısıyla vektörleri üretir (384 boyutlu).
- `vector_store/embeddings.npy` ve `vector_store/documents.jsonl` dosyalarını oluşturur.

`app.py` ilk çalıştırmada depo bulunamazsa aynı betiği otomatik çağırır. Başarısız olursa
uygulama “No-Vector” modunda Gemini’nin genel bilgisini kullanarak devam eder.

## 🚀 Uygulamayı Çalıştırma
```bash
streamlit run app.py
```

Arayüz:
- Gemini istemcisini yapılandırır (`configure_gemini`).
- Vektör deposundan bağlam çeker (`get_contexts` + `retrieve_with_threshold`).
- Sohbet geçmişi ve kurallarla prompt üretir (`build_prompt`).
- Yanıtı ve kullanılan kaynakları sohbet penceresinde listeler.

Yan çubuktan benzerlik eşikleri, fallback parametreleri, sıcaklık ve hafıza uzunluğu gibi
ayarları canlı olarak güncelleyebilirsiniz.

## 🛠️ Notlar & İpuçları
- Vektör deposunu sıfırlamak için `vector_store/` klasörünü silip ingest’i yeniden çalıştırın.
- Yerel embedding modeli ilk kez indirileceği için birkaç yüz MB’lık indirme sürebilir.
- `HF_DATASETS_OFFLINE=1` ile veri setlerini önceden indirilmiş cache’den kullanabilirsiniz.
- Streamlit, yeni mesaj geldiğinde otomatik olarak en son bağlamları getirir; manuel yenilemeye gerek yoktur.

## 🧰 Sorun Giderme
- `GOOGLE_API_KEY bulunamadı` hatası `.env` dosyasının yüklenmediğini gösterir; `load_dotenv()` çağrısı için çalışma dizininin doğru olduğundan emin olun.
- Hugging Face indirmeleri başarısız olursa betik uyarı verir ve eldeki verilerle devam eder; temiz bir başlangıç için tekrar çalıştırın.
- “No-Vector” modu uyarısı görürseniz `python ingest.py` komutunu manuel çalıştırarak ayrıntılı hatayı görebilirsiniz.

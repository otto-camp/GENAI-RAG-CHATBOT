# 💳 FinAI

Fintech odaklı dokümanları Google Gemini modelleriyle birleştiren hibrit bir
retrieval-augmented generation (RAG) sohbet uygulaması. ChromaDB üzerinde
kalıcı vektör hafızası tutar, uygun bağlam bulamazsa sohbeti kesmeden Gemini'nin
genel bilgisinden destek alır.

## ✨ Özellikler
- 💬 Streamlit tabanlı sohbet arayüzü (`app.py`)
- 🧠 Hibrit bağlam seçimi: sıkı eşik + fallback araması
- 🧵 Sohbet geçmişi (`max_history`) ve bağlamlara göre prompt inşası
- 🗂️ ChromaDB ile kalıcı vektör deposu
- 🧮 Embedding sağlayıcısı olarak Gemini (`text-embedding-004`) veya yerel
  `sentence-transformers` modeli
- 📚 HF veri setleri: `financial_phrasebank` ve `banking77`

## 🗃️ Proje Yapısı
```text
FinAI/
├─ app.py               # Streamlit sohbet arayüzü
├─ ingest.py            # Hugging Face verilerini ChromaDB'ye aktarma betiği
├─ rag_utils.py         # Gemini yapılandırması, embedding ve prompt yardımcıları
├─ requirements.txt     # Python bağımlılıkları
├─ .env.example         # Ortam değişkenleri için şablon
├─ chroma/              # Kalıcı vektör veritabanı (ingest sonrası oluşur)
└─ venv/                # (Opsiyonel) Proje sanal ortamı
```

## ⚙️ Kurulum
1. Projeyi klonlayın ve dizine girin.
2. Sanal ortam oluşturun:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows için: venv\Scripts\activate
   ```
3. Bağımlılıkları yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

## 🔐 Ortam Değişkenleri
1. `.env.example` dosyasını `.env` olarak kopyalayın:
   ```bash
   cp .env.example .env
   ```
2. Aşağıdaki alanları doldurun:
   - `GOOGLE_API_KEY`: Gemini API anahtarınız
   - `GOOGLE_LLM_MODEL`: Varsayılan `models/gemini-2.0-flash`
   - `GOOGLE_MAX_OUTPUT_TOKENS`: Yanıt uzunluğu sınırı (varsayılan 512)
   - Embed ayarları:
     - `EMBED_PROVIDER`: `local` (varsayılan) veya `gemini`
     - `HF_LIMIT`: Her veri setinden çekilecek maksimum doküman sayısı
     - `EMBED_BATCH_SIZE` ve `EMBED_SLEEP_BETWEEN`: Yerel/Gemini embedding
       sırasında hız limitini ayarlamak için

## 📥 Veri Hazırlama (Ingest)
ChromaDB deposunu doldurmak için:
```bash
python ingest.py
```

Betik Hugging Face üzerinden iki veri seti indirir, normalize eder ve seçilen
embedding sağlayıcısıyla vektörleştirip `chroma/` klasörüne kaydeder. İlk
çalıştırmada indirmeler internet bağlantısı gerektirir. `EMBED_PROVIDER=gemini`
seçilirse hız limitlerine takılmamak için API anahtarınızın geçerli olduğundan
emin olun.

## 🚀 Uygulamayı Çalıştırma
Streamlit arayüzünü başlatmak için:
```bash
streamlit run app.py
```

Uygulama:
- Google Gemini API'sini yapılandırır (`configure_gemini`)
- ChromaDB'den bağlamları çeker (`get_contexts`, `retrieve_with_threshold`)
- Sohbet geçmişi + bağlamlarla prompt oluşturur (`build_prompt`)
- Gemini'den yanıt üretir ve kullanılan kaynakları listeler

Yan çubuktaki ayarlarla benzerlik eşiği, fallback parametreleri, sıcaklık ve
hafıza uzunluğunu anlık değiştirebilirsiniz.

## 🛠️ Geliştirme Notları
- Chroma deposunu sıfırlamak için `chroma/` klasörünü silerek ingest'i tekrar
  çalıştırabilirsiniz.
- Yerel embedding modeli (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
  384 boyutlu vektörler üretir; ilk kullanımda model indirileceğinden biraz süre
  alabilir.
- `rag_utils.py` dosyası ortak yardımcıları (Gemini istemcisi, embedding
  fonksiyonları ve prompt kuralları) barındırır.

## 🧰 Sorun Giderme
- `GOOGLE_API_KEY bulunamadı` hatası alırsanız `.env` dosyasını kontrol edin ve
  uygulamayı yeniden başlatmadan önce `load_dotenv()` çağrısının anahtarı
  görebildiğinden emin olun.
- Hugging Face veri seti indirme hataları kritik değildir; betik kalan
  veri setleriyle devam eder. Daha fazla bağlam isterseniz komut satırında
  tekrar çalıştırabilirsiniz.
- Streamlit arayüzü bağlam bulamazsa yanıtın sonunda modelin genel bilgilerden
  yararlandığını belirten not eklenir; bu davranış `build_prompt` içinde
  kontrol edilir.

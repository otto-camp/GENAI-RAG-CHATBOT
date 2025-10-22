# app.py
import os
import sys
import subprocess
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google.genai import types
from streamlit.components.v1 import html as st_html  # modal/popup için

from rag_utils import (
    configure_gemini,
    get_model_config,
    embed_texts,
    build_prompt,
    load_vector_store,
    query_vector_store,
)

# ============== Setup ==============
st.set_page_config(page_title="FinAI", page_icon="💳", layout="wide")

THEME_PATH = Path(__file__).parent / "styles" / "finai_theme.css"


def inject_theme_css() -> None:
    try:
        css = THEME_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.warning("Tema dosyası bulunamadı; varsayılan Streamlit stilinde devam ediliyor.")
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


inject_theme_css()
load_dotenv()
client = configure_gemini()
MODEL_NAME, MAX_TOKENS = get_model_config()


# ============== Vector store loader (güvenli) ==============
@st.cache_resource(show_spinner=False)
def get_vector_store():
    try:
        return load_vector_store()
    except Exception as e:
        st.session_state["vector_load_error"] = str(e)
        return None


VECTOR_STORE = get_vector_store()

# Vektör deposu yoksa kullanıcıya aksiyon sun
if VECTOR_STORE is None:
    st.error("Vektör deposu bulunamadı. Önce embeddings oluşturmalısın.")
    with st.expander("Detay / Log"):
        st.write(st.session_state.get("vector_load_error", ""))

    col1, col2 = st.columns(2)
    with col1:
        run_ingest = st.button("📦 Vektör Deposunu Şimdi Oluştur (ingest.py)")
    with col2:
        st.markdown(
            "<div class='transparent-note'>Alternatif: Lokal ortamda <code>python ingest.py</code> çalıştırıp yeniden deploy et.</div>",
            unsafe_allow_html=True,
        )

    if run_ingest:
        with st.spinner("İçerikler işleniyor, embeddings oluşturuluyor..."):
            try:
                result = subprocess.run(
                    [sys.executable, "ingest.py"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.stdout:
                    st.code(result.stdout, language="bash")
                if result.returncode != 0:
                    st.error("ingest.py başarısız oldu. stderr:")
                    st.code(result.stderr or "", language="bash")
                else:
                    st.success("Ingest tamam! Vektör deposu yeniden yükleniyor...")
                    # cache'i temizle
                    get_vector_store.clear()
                    # yeniden yükleyip sayfayı tazele
                    _ = get_vector_store()
                    st.rerun()
            except FileNotFoundError:
                st.error("ingest.py dosyası bulunamadı. Yerelde `python ingest.py` çalıştırıp yeniden deploy et.")
            except Exception as e:
                st.error(f"Beklenmeyen hata: {e}")

    # vektör deposu olmadan devam etmeyelim
    st.stop()


# ============== Sidebar ==============
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-icon">💳</div>
        <div>
            <h1>FinAI</h1>
            <p>Fintech içgörüleri için bellekli, RAG destekli uzman sohbet asistanı.</p>
            <div class="hero-badges">
                <span class="hero-badge">Hybrid Retrieval</span>
                <span class="hero-badge">Vector Store</span>
                <span class="hero-badge">Gemini Models</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header">
            <h2>Yanıt Ayarları</h2>
            <p>FinAI'nin arama derinliğini ve üretim tarzını kişiselleştir.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("⚙️ Ayarlar")
    top_k = st.slider("Top K (ilk deneme)", 1, 10, 4)
    similarity_threshold = st.slider("Benzerlik eşiği", 0.30, 0.90, 0.55, 0.05)
    fallback_top_k = st.slider(
        "Top K (fallback)", 1, 12, 8, help="İlk denemede bağlam azsa ikinci deneme için."
    )
    fallback_similarity = st.slider("Eşik (fallback)", 0.20, 0.80, 0.35, 0.05)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_history = st.slider("Hafıza (son N mesaj)", 0, 12, 6)
    st.markdown("---")
    st.markdown(f"**Model:** `{MODEL_NAME}`")
    st.markdown(f"**Max tokens:** `{MAX_TOKENS}`")

# ============== Memory ==============
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# ============== Retrieval helpers ==============
def retrieve_with_threshold(query: str, k: int, min_similarity: float):
    # Emniyet: VECTOR_STORE zaten mevcut; yine de None kontrolü
    if VECTOR_STORE is None:
        return []
    q_emb = embed_texts(client, [query])[0]
    res = query_vector_store(VECTOR_STORE, q_emb, k)
    ctxs = []
    best_ctx = None
    for item in res:
        metadata = item.get("metadata") or {}
        similarity = float(item.get("similarity", 0.0))
        candidate = {
            "text": item.get("text") or "",
            "source": metadata.get("source"),
            "page": metadata.get("page", 1),
            "similarity": similarity,
            "metadata": metadata,
            "index": item.get("index"),
        }
        if similarity >= min_similarity:
            ctxs.append(candidate)
        if best_ctx is None or similarity > best_ctx["similarity"]:
            best_ctx = candidate
    if not ctxs and best_ctx:
        ctxs.append(best_ctx)
    ctxs.sort(key=lambda x: x["similarity"], reverse=True)
    return ctxs


def get_contexts(query: str):
    # 1) sıkı arama
    ctxs = retrieve_with_threshold(query, top_k, similarity_threshold)
    # 2) zayıfsa fallback (eşik düşür, k arttır)
    if len(ctxs) == 0:
        alt = retrieve_with_threshold(query, fallback_top_k, fallback_similarity)
        return alt, True
    return ctxs, False


# ============== Modal builder (Kaynak listesi + popup) ==============
def build_modal_html(ctxs):
    """
    ctxs listesini '1. Kaynak, 2. Kaynak ...' olarak gösterir.
    Her birine tıklayınca ayrıntıların olduğu modal (popup) açılır.
    """
    seen = set()
    links_html = []
    modals_html = []
    idx = 0

    for c in ctxs:
        key = (c.get("index"), c.get("source"))
        if key in seen:
            continue
        seen.add(key)
        idx += 1

        source_label = escape(c.get("source") or "Bilinmiyor")
        page = c.get("page", 1)
        similarity = float(c.get("similarity", 0.0))
        meta = c.get("metadata") or {}
        question_preview = (meta.get("question") or "").strip()
        answer_preview = (meta.get("answer") or "").strip()
        context_preview = (meta.get("context") or "").strip()
        raw_text = (c.get("text") or "").strip()

        snippet_parts = []
        if question_preview:
            snippet_parts.append(
                f"<div class='context-snippet'><strong>Soru:</strong> {escape(question_preview[:220])}</div>"
            )
        if answer_preview:
            snippet_parts.append(
                f"<div class='context-snippet'><strong>Cevap:</strong> {escape(answer_preview[:220])}</div>"
            )
        if context_preview:
            snippet_parts.append(
                f"<div class='context-snippet'><strong>Bağlam:</strong> {escape(context_preview[:220])}</div>"
            )
        if raw_text and not snippet_parts:
            snippet_parts.append(f"<div class='context-snippet'>{escape(raw_text[:220])}</div>")

        for label_key, label_title in (
            ("category", "Kategori"),
            ("language", "Dil"),
            ("pii_type", "PII"),
            ("product", "Ürün"),
            ("issue", "Konu"),
            ("company", "Şirket"),
        ):
            extra_val = (meta.get(label_key) or "").strip()
            if extra_val:
                snippet_parts.append(
                    f"<div class='context-snippet'><strong>{label_title}:</strong> {escape(extra_val[:220])}</div>"
                )

        snippet_html = "".join(snippet_parts)

        modal_id = f"modal-{idx}"
        links_html.append(
            f"<li><a href='#' class='src-link' data-target='{modal_id}'>{idx}. Kaynak</a></li>"
        )

        modals_html.append(
            f"""
        <div id="{modal_id}" class="modal">
          <div class="modal-content">
            <span class="close" data-target='{modal_id}'>&times;</span>
            <div class="context-title">{source_label}</div>
            <div class="context-meta">Benzerlik ≈ {similarity:.2f} · Sayfa {page}</div>
            <div class="snippet-wrap">{snippet_html or "<em>Detay bulunamadı</em>"}</div>
          </div>
        </div>
        """
        )

    full_html = f"""
    <style>
      ul.source-list {{ list-style: none; padding-left: 0; margin: 4px 0 10px; }}
      ul.source-list li {{ margin: 6px 0; }}
      .src-link {{ text-decoration: none; cursor: pointer; }}
      .src-link:hover {{ text-decoration: underline; }}

      .modal {{
        display: none;
        position: fixed; z-index: 9999; left: 0; top: 0;
        width: 100%; height: 100%; overflow: auto;
        background-color: rgba(0,0,0,0.45);
      }}
      .modal-content {{
        background-color: #fff;
        margin: 8% auto; padding: 16px 20px; border-radius: 8px;
        max-width: 760px; width: calc(100% - 40px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      }}
      .close {{
        float: right; font-size: 26px; font-weight: bold; cursor: pointer;
      }}
      .context-title {{ font-weight: 600; font-size: 18px; margin-bottom: 6px; }}
      .context-meta {{ color: #666; font-size: 13px; margin-bottom: 12px; }}
      .context-snippet {{ margin: 6px 0; line-height: 1.4; }}
      .snippet-wrap {{ max-height: 60vh; overflow-y: auto; }}
    </style>

    <ul class="source-list">
      {''.join(links_html) if links_html else "<li>Kaynak bulunamadı</li>"}
    </ul>

    {''.join(modals_html)}

    <script>
      (function(){{
        function openModal(id) {{
          var m = document.getElementById(id);
          if (m) m.style.display = 'block';
        }}
        function closeModal(id) {{
          var m = document.getElementById(id);
          if (m) m.style.display = 'none';
        }}
        document.querySelectorAll('.src-link').forEach(function(a){{
          a.addEventListener('click', function(e){{
            e.preventDefault();
            var id = this.getAttribute('data-target');
            openModal(id);
          }});
        }});
        document.querySelectorAll('.close').forEach(function(x){{
          x.addEventListener('click', function(){{
            var id = this.getAttribute('data-target');
            closeModal(id);
          }});
        }});
        window.addEventListener('click', function(e){{
          if (e.target.classList && e.target.classList.contains('modal')) {{
            e.target.style.display = 'none';
          }}
        }});
      }})();
    </script>
    """
    return full_html


# ============== Chat loop ==============
user_q = st.chat_input("Fintech hakkında sorunu yaz (örn: 'PSD2 nedir?')")

if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Uygun bağlam aranıyor, yanıt hazırlanıyor..."):
            ctxs, _ = get_contexts(user_q)

            if len(ctxs) == 0:
                answer = "📭 Bu soruya ait içerik veri setimizde bulunamadı."
            else:
                prompt = build_prompt(
                    user_q,
                    ctxs,
                    chat_history=st.session_state.messages,
                    max_history=max_history,
                    allow_general_knowledge=False,
                )

                resp = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=MAX_TOKENS,
                    ),
                )

                # sağlam text çıkarımı
                answer = None
                if getattr(resp, "text", None):
                    answer = resp.text
                elif getattr(resp, "candidates", None):
                    try:
                        parts = resp.candidates[0].content.parts
                        if parts and hasattr(parts[0], "text"):
                            answer = parts[0].text
                    except Exception:
                        pass
                if not answer:
                    answer = "⚠️ Modelden metin alınamadı, tekrar deneyebilirim."

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # ===== bağlam özeti (modal liste) =====
            st.markdown("**📚 Kullanılan veri seti bağlamları:**")
            if ctxs:
                html_blob = build_modal_html(ctxs)
                st_html(html_blob, height=360, scrolling=True)
            else:
                st.markdown(
                    """
                    <div class="transparent-note">
                        🔒 Bu başlık için veri setinden sonuç bulunamadı.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# rag_utils.py
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from google import genai
from google.genai import types

# ---------- Client & config ----------
def configure_gemini() -> genai.Client:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY bulunamadı (.env)")
    return genai.Client(api_key=key)

def get_model_config():
    model = os.getenv("GOOGLE_LLM_MODEL", "models/gemini-2.0-flash")
    max_tokens = int(os.getenv("GOOGLE_MAX_OUTPUT_TOKENS", "512"))
    return model, max_tokens

def get_embed_provider() -> str:
    return os.getenv("EMBED_PROVIDER", "local").lower()  # local | gemini

# ---------- Utils ----------
def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- Embedding ----------
_local_model_cache = None
def embed_texts(client: Optional[genai.Client], texts: List[str]) -> List[List[float]]:
    provider = get_embed_provider()
    if provider == "local":
        return _embed_local(texts)
    else:
        if client is None:
            raise RuntimeError("Gemini client gerekli (EMBED_PROVIDER=gemini).")
        return _embed_gemini(client, texts)

def _embed_local(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    global _local_model_cache
    if _local_model_cache is None:
        from sentence_transformers import SentenceTransformer
        # hızlı ve çok dilli (384-dim)
        _local_model_cache = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embs = _local_model_cache.encode(
        texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
    )
    return embs.tolist()

def _embed_gemini(client: genai.Client, texts: List[str], batch_size: int = 8, sleep_between: float = 0.2) -> List[List[float]]:
    embs: List[List[float]] = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        for t in batch:
            for attempt in range(3):
                try:
                    resp = client.models.embed_content(model="text-embedding-004", contents=t)
                    embs.append(resp.embeddings[0].values)
                    break
                except Exception:
                    time.sleep(0.4 * (attempt + 1))
        if sleep_between > 0:
            time.sleep(sleep_between)
    return embs

# ---------- Vector store ----------
VECTOR_STORE_DIR = Path(__file__).parent / "vector_store"
_store_cache: Optional[Dict[str, Any]] = None


def load_vector_store() -> Dict[str, Any]:
    global _store_cache
    if _store_cache is not None:
        return _store_cache

    embeddings_path = VECTOR_STORE_DIR / "embeddings.npy"
    meta_path = VECTOR_STORE_DIR / "documents.jsonl"
    if not embeddings_path.exists() or not meta_path.exists():
        raise RuntimeError(
            f"Vektör deposu bulunamadı. Lütfen önce `python ingest.py` çalıştır (beklenen dosya: {embeddings_path})."
        )

    embeddings = np.load(embeddings_path, mmap_mode="r").astype(np.float32)
    documents: List[Dict[str, Any]] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            documents.append({
                "text": entry.get("text", ""),
                "meta": entry.get("meta") or {},
            })

    if embeddings.shape[0] != len(documents):
        raise RuntimeError(
            f"Gömme ve doküman sayısı eşleşmiyor: {embeddings.shape[0]} vs {len(documents)}."
        )

    _store_cache = {"embeddings": embeddings, "documents": documents}
    return _store_cache


def query_vector_store(store: Dict[str, Any], query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
    if top_k <= 0:
        return []
    embeddings: np.ndarray = store["embeddings"]
    if embeddings.size == 0:
        return []
    documents: List[Dict[str, Any]] = store["documents"]

    query_vec = np.asarray(query_embedding, dtype=np.float32)
    norm = np.linalg.norm(query_vec)
    if norm == 0:
        return []
    query_vec = query_vec / norm

    sims = embeddings @ query_vec
    k = min(top_k, sims.shape[0])
    if k <= 0:
        return []
    top_idx = np.argpartition(-sims, k - 1)[:k]
    sorted_idx = top_idx[np.argsort(-sims[top_idx])]

    results: List[Dict[str, Any]] = []
    for idx in sorted_idx:
        doc_entry = documents[int(idx)]
        meta = doc_entry.get("meta") or {}
        similarity = float(sims[int(idx)])
        results.append({
            "text": doc_entry.get("text", ""),
            "metadata": meta,
            "similarity": similarity,
            "index": int(idx),
        })
    return results

# ---------- Prompt (hybrid RAG + memory) ----------
def build_prompt(
    user_q: str,
    contexts: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
    max_history: int = 6,
    allow_general_knowledge: bool = True,
) -> str:
    # history
    history_text = ""
    if chat_history:
        for msg in chat_history[-max_history:]:
            role = "Kullanıcı" if msg["role"] == "user" else "Asistan"
            history_text += f"{role}: {msg['content']}\n"

    # contexts
    if contexts:
        ctx_text = "\n\n---\n\n".join(
            [f"[Kaynak: {c.get('source','?')}] \n{c['text']}" for c in contexts]
        )
    else:
        ctx_text = "(Bu turda uygun bağlam bulunamadı.)"

    # rules
    if allow_general_knowledge:
        rules = (
            "Aşağıda finans alanına ait veri setlerinden çekilmiş bağlamlar ve sohbet geçmişi var. "
            "Önce bağlamı kullanarak cevap ver ve kısa tut (en fazla üç cümle). "
            "Eğer bağlam yetersizse kendi genel bilginle tamamlayabilirsin; ancak bağlamdaki bilgilerle çelişme. "
            "Yanıtı kullanıcının soru sorduğu dilde ver; dil emin değilse Türkçe yanıt kullan. "
            "Başlık, madde işaretleri veya takip sorusu ekleme."
        )
    else:
        rules = (
            "Aşağıda finans veri setlerinden çekilen bağlamlar ve sohbet geçmişi var. "
            "Yanıtını yalnızca bağlamdaki bilgiye dayandır ve kısa tut (en fazla üç cümle). "
            "Yanıtı kullanıcının soru sorduğu dilde ver; dil emin değilse Türkçe yanıt kullan. "
            "Başlık, madde işaretleri veya takip sorusu ekleme. "
            "Bağlam soru için yeterli değilse sadece 'Bu konuda veri setinde bilgi bulunamadı.' yaz."
        )

    return (
        f"{rules}\n\n"
        f"Sohbet Geçmişi:\n{history_text}\n\n"
        f"Bağlam:\n{ctx_text}\n\n"
        f"Soru: {user_q}\n\n"
        f"Cevap:"
    )

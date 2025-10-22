# ingest.py
import os
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from tqdm import tqdm

from rag_utils import configure_gemini, embed_texts, normalize_text, get_embed_provider

# -----------------------------
# Hugging Face dataset okuma
# -----------------------------
Record = Dict[str, Optional[str]]
Extractor = Callable[[Dict[str, Any]], Optional[Record]]

CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "datasets"


def _match_child_dir(parent: Path, name: str) -> Path:
    target = name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    raise FileNotFoundError(f"{parent} altında {name} klasörü bulunamadı.")


def _iter_cached_rows(hf_id: str, split: str, config: Optional[str], limit: Optional[int]):
    dataset_dir = _match_child_dir(CACHE_ROOT, hf_id.replace("/", "___"))
    config_dir = _match_child_dir(dataset_dir, config or "default")
    version_dirs = [p for p in config_dir.iterdir() if p.is_dir()]
    if not version_dirs:
        raise FileNotFoundError(f"{config_dir} altında versiyon klasörü bulunamadı.")
    version_dirs.sort(key=lambda p: p.name)
    arrow_paths = []
    for vdir in reversed(version_dirs):
        arrow_paths.extend(sorted(vdir.glob(f"**/*{split}*.arrow")))
        if arrow_paths:
            break
    if not arrow_paths:
        raise FileNotFoundError(f"{hf_id} için {split} arrow dosyası bulunamadı ({config_dir}).")
    yielded = 0
    for path in arrow_paths:
        ds = Dataset.from_file(str(path))
        for row in ds:
            yield row
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def _coalesce_text(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            candidate = value.strip()
        elif isinstance(value, (list, tuple, set)):
            candidate = " ".join(str(v) for v in value if v)
        elif isinstance(value, dict):
            candidate = " ".join(str(v) for v in value.values() if v)
        else:
            candidate = str(value)
        candidate = candidate.strip()
        if candidate:
            return candidate
    return None


def _extract_finance_instruct(row: Dict[str, Any]) -> Optional[Record]:
    instruction = _coalesce_text(
        row.get("user"),
        row.get("instruction"),
        row.get("prompt"),
        row.get("question"),
        row.get("task"),
    )
    input_text = _coalesce_text(
        row.get("system"),
        row.get("input"),
        row.get("context"),
        row.get("details"),
        row.get("background"),
    )
    response = _coalesce_text(
        row.get("assistant"),
        row.get("output"),
        row.get("response"),
        row.get("answer"),
        row.get("completion"),
    )
    if not instruction or not response:
        return None
    category = _coalesce_text(
        row.get("category"),
        row.get("sub_category"),
        row.get("domain"),
        row.get("topic"),
        row.get("type"),
    )
    difficulty = _coalesce_text(row.get("difficulty"), row.get("complexity"))
    parts: List[str] = [f"Talimat: {instruction}"]
    if input_text:
        parts.append(f"Girdi: {input_text}")
    parts.append(f"Cevap: {response}")
    record: Record = {
        "text": "\n".join(parts),
        "question": instruction,
        "answer": response,
        "context": input_text,
        "category": category,
        "difficulty": difficulty,
        "source_ref": _coalesce_text(row.get("source"), row.get("dataset"), row.get("reference")),
    }
    return record


def _extract_gretel_finance(row: Dict[str, Any]) -> Optional[Record]:
    document_text = _coalesce_text(
        row.get("generated_text"),
        row.get("document"),
        row.get("text"),
    )
    if not document_text:
        return None
    description = _coalesce_text(
        row.get("document_description"),
        row.get("expanded_description"),
        row.get("domain"),
    )
    prompt = _coalesce_text(
        row.get("prompt"),
        row.get("instruction"),
        row.get("question"),
        description,
    )
    completion = document_text
    context = _coalesce_text(
        row.get("expanded_type"),
        row.get("document_type"),
        row.get("language_description"),
    )
    language = _coalesce_text(row.get("language"))
    pii_labels = _coalesce_text(
        row.get("pii_types"),
        row.get("pii_entities"),
        row.get("pii_entity"),
        row.get("pii_type"),
    )
    parts: List[str] = []
    if prompt:
        parts.append(f"Açıklama: {prompt}")
    if context:
        parts.append(f"Bağlam: {context}")
    parts.append(f"Metin: {completion}")
    question = prompt or context or "Finans belgesi"
    record: Record = {
        "text": "\n".join(parts),
        "question": question,
        "answer": completion,
        "context": context,
        "language": language,
        "pii_type": pii_labels,
        "contains_pii": _coalesce_text(row.get("contains_pii")),
        "document_type": _coalesce_text(row.get("document_type")),
        "document_description": description,
    }
    return record


def _extract_cfpb(row: Dict[str, Any]) -> Optional[Record]:
    complaint = _coalesce_text(
        row.get("consumer_complaint_narrative"),
        row.get("complaint_what_happened"),
        row.get("issue_description"),
    )
    response = _coalesce_text(
        row.get("company_public_response"),
        row.get("company_response_to_consumer"),
        row.get("company_response"),
    )
    if not complaint or not response:
        return None
    product = _coalesce_text(row.get("product"))
    sub_product = _coalesce_text(row.get("sub_product"))
    issue = _coalesce_text(row.get("issue"))
    sub_issue = _coalesce_text(row.get("sub_issue"))
    parts: List[str] = [f"Şikayet: {complaint}"]
    summary_bits = []
    if issue:
        summary_bits.append(f"Konu: {issue}")
    if product:
        summary_bits.append(f"Ürün: {product}")
    if summary_bits:
        parts.append(" · ".join(summary_bits))
    parts.append(f"Şirket Yanıtı: {response}")
    record: Record = {
        "text": "\n".join(parts),
        "question": complaint,
        "answer": response,
        "context": issue,
        "product": product,
        "sub_product": sub_product,
        "issue": issue,
        "sub_issue": sub_issue,
        "company": _coalesce_text(row.get("company"), row.get("company_name")),
        "state": _coalesce_text(row.get("state")),
        "date_received": _coalesce_text(row.get("date_received")),
    }
    return record


DATASET_SPECS: List[Dict[str, Any]] = [
    {
        "hf_id": "Josephgflowers/Finance-Instruct-500k",
        "split": "train",
        "label": "finance_instruct",
        "extractor": _extract_finance_instruct,
    },
    {
        "hf_id": "gretelai/synthetic_pii_finance_multilingual",
        "split": "train",
        "label": "synthetic_pii_finance",
        "extractor": _extract_gretel_finance,
    },
]

OPTIONAL_DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "cfpb": {
        "hf_id": "CFPB/consumer-finance-complaints",
        "split": "train",
        "label": "cfpb_complaints",
        "extractor": _extract_cfpb,
    }
}


def read_hf_datasets(limit_per_ds: Optional[int], include_cfpb: bool) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    specs = list(DATASET_SPECS)
    if include_cfpb:
        specs.append(OPTIONAL_DATASET_SPECS["cfpb"])

    for spec in specs:
        hf_id = spec["hf_id"]
        label = spec["label"]
        split = spec.get("split", "train")
        config = spec.get("config")
        extractor: Extractor = spec["extractor"]
        limit_override = spec.get("limit")
        limit = limit_override if limit_override is not None else limit_per_ds

        print(f"🔹 HF: {hf_id} ({label}) yükleniyor...")
        rows_iter: Optional[Any] = None
        if os.getenv("HF_DATASETS_OFFLINE", "0") == "1":
            ds = None
        else:
            try:
                if config:
                    ds = load_dataset(hf_id, config, split=split)
                else:
                    ds = load_dataset(hf_id, split=split)
                rows_iter = iter(ds)
            except Exception as e:
                print(f"[WARN] {hf_id} alınamadı: {e}")
        if rows_iter is None:
            try:
                rows_iter = _iter_cached_rows(hf_id, split, config, limit)
                print(f"   ↳ Önbellekten okunuyor ({label}).")
            except Exception as cache_err:
                print(f"[WARN] {hf_id} önbellekten alınamadı: {cache_err}")
                continue

        cnt = 0
        for row in rows_iter:
            extracted = extractor(row)
            if not extracted:
                continue
            text = normalize_text(extracted.get("text") or "")
            if not text:
                continue
            meta = {k: v for k, v in extracted.items() if k != "text" and v}
            meta["source"] = label
            docs.append({"source": label, "text": text, "meta": meta})
            cnt += 1
            if limit is not None and cnt >= limit:
                break
        print(f"   ↳ {cnt} kayıt eklendi ({label}).")

    print(f"✅ Toplam {len(docs)} metin alındı (dataset sayısı={len(specs)}).")
    return docs

# -----------------------------
# Ana akış
# -----------------------------
def main():
    load_dotenv()

    hf_limit = int(os.getenv("HF_LIMIT", "1000"))       # her dataset için üst sınır (0 veya negatif = limitsiz)
    limit_value = None if hf_limit <= 0 else hf_limit
    provider = get_embed_provider()                      # local | gemini
    client = configure_gemini() if provider == "gemini" else None
    include_cfpb = os.getenv("INCLUDE_CFPB_DATASET", "false").lower() in {"1", "true", "yes"}

    raw_docs = read_hf_datasets(limit_per_ds=limit_value, include_cfpb=include_cfpb)
    if not raw_docs:
        print("❌ Hiç veri yok.")
        return

    # de-dup
    seen = set()
    docs, metas = [], []
    for i, d in enumerate(raw_docs):
        t = d["text"]
        if not t or t in seen:
            continue
        seen.add(t)
        docs.append(t)
        meta = dict(d.get("meta") or {})
        if "source" not in meta:
            meta["source"] = d["source"]
        metas.append(meta)

    print(f"🧮 Embed edilecek metin sayısı: {len(docs)} (provider={provider})")
    embs = embed_texts(client, docs)

    emb_array = np.asarray(embs, dtype=np.float32)
    if emb_array.ndim != 2:
        raise RuntimeError(f"Gömme boyutu beklenmedik: {emb_array.shape}")
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_array = emb_array / norms

    store_dir = (Path(__file__).parent / "vector_store").resolve()
    store_dir.mkdir(parents=True, exist_ok=True)
    emb_path = store_dir / "embeddings.npy"
    meta_path = store_dir / "documents.jsonl"

    np.save(emb_path, emb_array)
    with meta_path.open("w", encoding="utf-8") as f:
        for text, meta in tqdm(zip(docs, metas), total=len(docs), unit="doc", desc="📄 Kayıt yazılıyor"):
            json.dump({"text": text, "meta": meta}, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Vektör deposu hazırlandı: {emb_path.relative_to(Path(__file__).parent)} & {meta_path.relative_to(Path(__file__).parent)}")

if __name__ == "__main__":
    main()

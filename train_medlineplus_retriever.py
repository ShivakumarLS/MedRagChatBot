"""
Train & save a medical-domain embedding model + FAISS index using MedlinePlus XML.

What you get:
- trained embedding model (SentenceTransformers)
- faiss index of all MedlinePlus topic documents
- docs metadata mapping (json)

Install:
  pip install -U sentence-transformers faiss-cpu datasets lxml requests tqdm

Run:
  python train_medlineplus_retriever.py
"""

import os
import re
import json
import zipfile
import shutil
import requests
from tqdm import tqdm
from lxml import etree

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.datasets import SentencesDataset
from torch.utils.data import DataLoader


# ----------------------------
# Config
# ----------------------------
OUT_DIR = "rag_artifacts"
DATA_DIR = os.path.join(OUT_DIR, "data")
MODEL_DIR = os.path.join(OUT_DIR, "trained_retriever_model")
INDEX_PATH = os.path.join(OUT_DIR, "faiss.index")
DOCS_JSON = os.path.join(OUT_DIR, "docs.json")

# A strong starter embedding model (fast + good quality)
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# MedlinePlus Health Topic XML (compressed .zip). Official page: https://medlineplus.gov/xml.html
# We'll fetch the HTML and discover the latest "Compressed Health Topic XML" link automatically.
MEDLINEPLUS_XML_PAGE = "https://medlineplus.gov/xml.html"

# Training
BATCH_SIZE = 64
EPOCHS = 1
LR_WARMUP_STEPS = 100
MAX_TRAIN_PAIRS = 40000   # cap to keep training quick; raise if you want
EMBED_BATCH = 128


# ----------------------------
# Helpers
# ----------------------------
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s


def download_file(url: str, out_path: str):
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(out_path)) as pbar:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def find_compressed_xml_zip_link() -> str:
    """
    MedlinePlus XML page contains links; we pick the 'Compressed Health Topic XML' .zip.
    """
    html = requests.get(MEDLINEPLUS_XML_PAGE, timeout=60).text
    # pick first .zip that looks like health topic XML
    # (MedlinePlus sometimes changes filenames; this keeps it resilient.)
    candidates = re.findall(r'href="([^"]+\.zip)"', html, flags=re.IGNORECASE)
    if not candidates:
        raise RuntimeError("Could not find any .zip link on MedlinePlus XML page.")
    # Prefer ones that look like health topics
    priority = [c for c in candidates if "healthtopics" in c.lower() or "topic" in c.lower()]
    link = (priority[0] if priority else candidates[0])
    if link.startswith("/"):
        link = "https://medlineplus.gov" + link
    elif link.startswith("http") is False:
        link = "https://medlineplus.gov/" + link.lstrip("./")
    return link


def extract_zip(zip_path: str, out_dir: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
        # find the largest xml inside
        xml_files = [os.path.join(out_dir, n) for n in z.namelist() if n.lower().endswith(".xml")]
    if not xml_files:
        raise RuntimeError("No XML file found in the downloaded zip.")
    xml_files.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return xml_files[0]


def parse_medlineplus_topics(xml_path: str):
    """
    Parse MedlinePlus health topic XML into a list of documents.
    We build one document per topic with: title + summary + selected sections (if present).
    """
    docs = []
    context = etree.iterparse(xml_path, events=("end",), recover=True, huge_tree=True)

    for _, elem in tqdm(context, desc="Parsing XML"):
        tag = elem.tag.lower()
        if tag.endswith("healthtopic"):
            # Title
            title = elem.findtext(".//healthtopicname") or elem.findtext(".//title") or ""
            title = clean_text(title)

            # Summary (common field)
            summary = elem.findtext(".//fullsummary") or elem.findtext(".//summary") or ""
            summary = clean_text(summary)

            # Optional sections (may vary by XML schema version)
            also_called = clean_text(elem.findtext(".//also-called") or "")
            group = clean_text(elem.findtext(".//groupname") or "")

            parts = [p for p in [f"Title: {title}", f"Summary: {summary}", f"Also called: {also_called}", f"Group: {group}"] if len(p) > 10]
            text = clean_text(" ".join(parts))

            if title and text:
                docs.append({
                    "id": f"medlineplus::{len(docs)}",
                    "title": title,
                    "text": text
                })

            # Free memory
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

    return docs


def build_training_pairs(docs):
    """
    Create (query, passage) pairs for contrastive training.
    Simple but effective approach:
      query = title
      positive = full topic text
    """
    pairs = []
    for d in docs:
        q = clean_text(d["title"])
        p = clean_text(d["text"])
        if len(q) >= 5 and len(p) >= 30:
            pairs.append(InputExample(texts=[q, p]))

    # cap size
    if len(pairs) > MAX_TRAIN_PAIRS:
        pairs = pairs[:MAX_TRAIN_PAIRS]
    return pairs


def train_and_save_model(train_examples):
    model = SentenceTransformer(BASE_MODEL)

    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    train_loss = MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=LR_WARMUP_STEPS,
        output_path=MODEL_DIR,
        show_progress_bar=True
    )
    return SentenceTransformer(MODEL_DIR)


def build_and_save_faiss(model, docs):
    texts = [d["text"] for d in docs]

    embeddings = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH), desc="Embedding docs"):
        batch = texts[i:i+EMBED_BATCH]
        emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        embeddings.append(emb)
    emb = np.vstack(embeddings).astype("float32")

    # cosine similarity via inner product on normalized vectors
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_JSON, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"\nSaved model to: {MODEL_DIR}")
    print(f"Saved FAISS index to: {INDEX_PATH}")
    print(f"Saved docs mapping to: {DOCS_JSON}")
    print(f"Total docs indexed: {len(docs)}")


def main():
    ensure_dirs()

    # 1) Download latest MedlinePlus compressed Health Topic XML zip
    zip_url = find_compressed_xml_zip_link()
    zip_path = os.path.join(DATA_DIR, "medlineplus_healthtopics.zip")
    print(f"Downloading: {zip_url}")
    download_file(zip_url, zip_path)

    # 2) Extract XML
    extracted_dir = os.path.join(DATA_DIR, "extracted")
    if os.path.exists(extracted_dir):
        shutil.rmtree(extracted_dir)
    os.makedirs(extracted_dir, exist_ok=True)

    xml_path = extract_zip(zip_path, extracted_dir)
    print(f"Using XML: {xml_path}")

    # 3) Parse topics into docs
    docs = parse_medlineplus_topics(xml_path)
    print(f"Parsed docs: {len(docs)}")

    # 4) Build training pairs and fine-tune embedding model
    train_examples = build_training_pairs(docs)
    print(f"Training pairs: {len(train_examples)}")

    trained_model = train_and_save_model(train_examples)

    # 5) Build FAISS index and save artifacts
    build_and_save_faiss(trained_model, docs)


if __name__ == "__main__":
    main()

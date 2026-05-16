# build_index.py
# MedlinePlus Topics XML -> Chunk -> Embeddings -> FAISS IndexFlatL2
#
# Install:
#   pip install -U requests lxml numpy faiss-cpu sentence-transformers
#
# Run:
#   python build_index.py

import os
import re
import json
import zipfile
import shutil
import html
import requests
import numpy as np
import faiss
from lxml import etree
from sentence_transformers import SentenceTransformer

# ----------------------------
# Config
# ----------------------------
MEDLINEPLUS_XML_PAGE = "https://medlineplus.gov/xml.html"  # official download page
OUT_DIR = "vectorstore"
DATA_DIR = "data"
INDEX_PATH = os.path.join(OUT_DIR, "faiss_index.bin")
META_PATH = os.path.join(OUT_DIR, "docs.json")

# Embeddings
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# ----------------------------
# Helpers
# ----------------------------
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

def clean_text(s: str) -> str:
    s = (s or "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def strip_html_escaped(s: str) -> str:
    """
    <full-summary> content is often HTML-escaped. Convert entities, remove tags.
    """
    s = html.unescape(s or "")
    s = re.sub(r"<[^>]+>", " ", s)  # drop tags
    s = re.sub(r"\s+", " ", s).strip()
    return s

def download_file(url: str, out_path: str):
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

def find_compressed_zip_url() -> str:
    """
    Find 'mplus_topics_compressed_YYYY-MM-DD.zip' from https://medlineplus.gov/xml.html
    """
    html_page = requests.get(MEDLINEPLUS_XML_PAGE, timeout=60).text
    zips = re.findall(r'href="([^"]+mplus_topics_compressed[^"]+\.zip)"', html_page, flags=re.IGNORECASE)

    if not zips:
        # fallback: any zip
        zips = re.findall(r'href="([^"]+\.zip)"', html_page, flags=re.IGNORECASE)

    if not zips:
        raise RuntimeError("Could not find any ZIP link on MedlinePlus XML page.")

    link = zips[0]
    if link.startswith("/"):
        link = "https://medlineplus.gov" + link
    elif not link.startswith("http"):
        link = "https://medlineplus.gov/" + link.lstrip("./")
    return link

def extract_zip(zip_path: str, out_dir: str) -> str:
    """
    Extract ZIP and return the largest XML file path.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
        xmls = [os.path.join(out_dir, n) for n in z.namelist() if n.lower().endswith(".xml")]

    if not xmls:
        raise RuntimeError("No XML found in downloaded zip.")

    xmls_sorted = sorted(xmls, key=lambda p: os.path.getsize(p), reverse=True)
    print("XML files extracted (largest first):")
    for p in xmls_sorted[:5]:
        print(" -", p, "size=", os.path.getsize(p))
    return xmls_sorted[0]

def debug_scan_tags(xml_path, n=80):
    """
    One-time scan: prints first ~80 unique tags (local names) to help debug schema.
    """
    seen = []
    seen_set = set()
    for _, elem in etree.iterparse(xml_path, events=("start",), recover=True, huge_tree=True):
        tag = elem.tag
        t = tag.split("}")[-1] if "}" in tag else tag
        if t not in seen_set:
            seen.append(t)
            seen_set.add(t)
        if len(seen) >= n:
            break
    print(f"Sample tags found (first ~{n} unique):")
    for t in seen:
        print(" -", t)

def chunk_text(text: str, chunk_size=900, overlap=150):
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks

# ----------------------------
# ✅ Correct parser for your XML
# ----------------------------
def parse_medlineplus(xml_path: str):
    """
    MedlinePlus Topics XML (like mplus_topics_YYYY-MM-DD.xml):
    - <health-topic> uses ATTRIBUTES for title/url/meta-desc (and others)
    - <full-summary> element contains HTML-escaped text
    """

    docs = []

    # We parse only health-topic end elements (namespace-safe + non-namespaced)
    context = etree.iterparse(
        xml_path,
        events=("end",),
        tag=("health-topic", "{*}health-topic"),
        recover=True,
        huge_tree=True
    )

    for _, elem in context:
        # ✅ title is an attribute
        title = clean_text(elem.get("title", ""))
        url = clean_text(elem.get("url", ""))
        meta_desc = clean_text(elem.get("meta-desc", ""))

        # ✅ full-summary element text (HTML-escaped)
        full_summary_raw = elem.findtext("full-summary") or ""
        summary = strip_html_escaped(full_summary_raw)

        # optional fields (may repeat)
        also_called = " ".join([clean_text((x.text or "")) for x in elem.findall("also-called") if clean_text(x.text or "")])
        groups = " | ".join([clean_text("".join(g.itertext())) for g in elem.findall("group") if clean_text("".join(g.itertext()))])

        # If summary empty, fallback to meta-desc
        if not summary:
            summary = meta_desc

        text = clean_text(" ".join([t for t in [
            f"Title: {title}" if title else "",
            f"Meta: {meta_desc}" if meta_desc else "",
            f"Summary: {summary}" if summary else "",
            f"Also called: {also_called}" if also_called else "",
            f"Groups: {groups}" if groups else "",
            f"URL: {url}" if url else "",
        ] if t]))

        if title and len(text) > 80:
            docs.append({"title": title, "text": text})

        # free memory
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    return docs

# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dirs()

    zip_url = find_compressed_zip_url()
    zip_path = os.path.join(DATA_DIR, "medlineplus_healthtopics.zip")
    print("Downloading MedlinePlus ZIP:", zip_url)
    download_file(zip_url, zip_path)

    extracted = os.path.join(DATA_DIR, "extracted")
    if os.path.exists(extracted):
        shutil.rmtree(extracted)
    os.makedirs(extracted, exist_ok=True)

    xml_path = extract_zip(zip_path, extracted)
    debug_scan_tags(xml_path)

    print("Parsing XML:", xml_path)
    raw_docs = parse_medlineplus(xml_path)

    print("Topics:", len(raw_docs))
    print("Sample titles:", [d["title"] for d in raw_docs[:5]])

    if not raw_docs:
        raise RuntimeError(
            "Parsed 0 topics from XML. The XML structure may be different than expected.\n"
            "If this happens, paste the first 30 lines of the XML so we can adjust attribute/tag names."
        )

    # Chunk
    chunked = []
    for d in raw_docs:
        chunks = chunk_text(d["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, ch in enumerate(chunks):
            chunked.append({
                "id": f"{d['title']}__{i}",
                "title": d["title"],
                "text": ch
            })

    print("Chunks:", len(chunked))
    if not chunked:
        raise RuntimeError("0 chunks created (unexpected).")

    # Embed
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode([c["text"] for c in chunked], convert_to_numpy=True, show_progress_bar=True)
    vectors = vectors.astype("float32")

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise RuntimeError(f"Embedding failed / empty vectors. vectors.shape={vectors.shape}")

    # FAISS IndexFlatL2 (paper uses IndexFlatL2)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Save
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunked, f, ensure_ascii=False, indent=2)

    print("Saved:", INDEX_PATH)
    print("Saved:", META_PATH)

if __name__ == "__main__":
    main()

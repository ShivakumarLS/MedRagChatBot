# eval_retrieval.py
import os, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict

VECTOR_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTOR_DIR, "faiss_index.bin")
META_PATH = os.path.join(VECTOR_DIR, "docs.json")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def main(k_list=(1,3,5,10)):
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise RuntimeError("Run build_index.py first (vectorstore missing).")

    index = faiss.read_index(INDEX_PATH)
    docs = json.load(open(META_PATH, "r", encoding="utf-8"))

    # Build gold mapping: title -> list of doc indices belonging to that title
    title_to_ids = defaultdict(list)
    for i, d in enumerate(docs):
        title_to_ids[d["title"]].append(i)

    titles = list(title_to_ids.keys())
    embedder = SentenceTransformer(EMBED_MODEL)

    recalls = {k: [] for k in k_list}
    mrrs = {k: [] for k in k_list}

    for title in titles:
        query = title
        gold = set(title_to_ids[title])

        qv = embedder.encode([query], convert_to_numpy=True).astype("float32")
        max_k = max(k_list)
        _, idxs = index.search(qv, max_k)
        retrieved = [int(x) for x in idxs[0]]

        for k in k_list:
            topk = retrieved[:k]
            hit = any(i in gold for i in topk)
            recalls[k].append(1.0 if hit else 0.0)

            # MRR@k
            rr = 0.0
            for rank, doc_id in enumerate(topk, start=1):
                if doc_id in gold:
                    rr = 1.0 / rank
                    break
            mrrs[k].append(rr)

    print("=== Retrieval Evaluation (title->chunks) ===")
    for k in k_list:
        print(f"Recall@{k}: {np.mean(recalls[k]):.4f} | MRR@{k}: {np.mean(mrrs[k]):.4f}")
    print(f"Queries: {len(titles)}")

if __name__ == "__main__":
    main()

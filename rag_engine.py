# rag_engine.py
import os, json, time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

VECTOR_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTOR_DIR, "faiss_index.bin")
META_PATH = os.path.join(VECTOR_DIR, "docs.json")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ✅ Set your local GGUF path here
GGUF_MODEL_PATH = os.path.join("models", "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf")

class RAGEngine:
    def __init__(self, top_k=4,
                 n_ctx=4096,
                 n_threads=None,
                 n_batch=256):
        if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
            raise RuntimeError("Vectorstore not found. Run: python build_index.py")

        if not os.path.exists(GGUF_MODEL_PATH):
            raise RuntimeError(f"GGUF model not found at: {GGUF_MODEL_PATH}")

        self.top_k = top_k
        self.embedder = SentenceTransformer(EMBED_MODEL)

        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.docs = json.load(f)

        # ✅ Offline LLM (GGUF)
        self.llm = Llama(
            model_path=GGUF_MODEL_PATH,
            n_ctx=n_ctx,
            n_threads=n_threads,  # None = auto
            n_batch=n_batch,
            verbose=False,
        )

    def retrieve(self, query: str):
        qv = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        distances, idxs = self.index.search(qv, self.top_k)
        results = []
        for i in idxs[0]:
            i = int(i)
            if 0 <= i < len(self.docs):
                results.append(self.docs[i])
        print("results==",results)
        return results

    def build_prompt(self, query: str, retrieved):
        context = "\n\n".join([f"[{d['title']}]\n{d['text']}" for d in retrieved])

        prompt = f"""
You are a medical information chatbot.
Use ONLY the provided context from MedlinePlus-like knowledge.
If the answer is not present in the context, say you don't know and suggest consulting a clinician.

User question: {query}

Context:
{context}

Write a helpful answer. Use bullet points when appropriate.
Also include a short safety note: "This is informational, not a diagnosis."
""".strip()

        return prompt

    def generate(self, prompt: str):
        # Llama.cpp "chat" style without needing HF pipeline
        out = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.3,
            top_p=0.9,
            stop=["</s>", "User question:"]
        )
        return (out["choices"][0]["text"] or "").strip()

    def answer(self, query: str):
        retrieved = self.retrieve(query)
        print("retrieved==",retrieved)
        prompt = self.build_prompt(query, retrieved)
        print("prompt==",prompt)
        reply = self.generate(prompt)
        print("reply===",reply)

        sources = list(dict.fromkeys([d["title"] for d in retrieved]))
        return reply, sources
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_data():
    dataset = load_dataset(
        "keivalya/MedQuad-MedicalQnADataset",
        split="train"
    ).select(range(800))

    texts = []

    for item in dataset:
        q = item.get("Question", "")
        a = item.get("Answer", "")

        if q and a and len(a) > 30:
            texts.append(f"Q: {q}\nA: {a}")

    print(f"📄 Documents loaded: {len(texts)}")

    # ---------------- EMBEDDINGS ----------------
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")

    # ---------------- BM25 ----------------
    tokenized_docs = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized_docs)

    return embeddings, texts, bm25
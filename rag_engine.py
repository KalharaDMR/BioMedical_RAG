import os
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class RAGEngine:
    def __init__(self, vector_store, texts, bm25):
        self.vs = vector_store
        self.texts = texts
        self.bm25 = bm25

    # ---------------- EMBED ----------------
    def embed(self, text):
        return embedding_model.encode(
            text,
            normalize_embeddings=True
        ).astype("float32")

    # ---------------- HYBRID RETRIEVAL ----------------
    def retrieve(self, query):

        # 1. VECTOR SEARCH (Qdrant)
        q_vec = self.embed(query)
        vector_results = self.vs.search(q_vec, top_k=5)
        vector_texts = [r.payload["text"] for r in vector_results]

        # 2. BM25 SEARCH
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        top_bm25 = np.argsort(bm25_scores)[-5:]

        bm25_texts = [self.texts[i] for i in top_bm25]

        # 3. MERGE RESULTS
        combined = list(set(vector_texts + bm25_texts))

        return combined[:8]

    # ---------------- GENERATION ----------------
    def generate(self, query):

        docs = self.retrieve(query)

        context = "\n\n".join(docs)

        prompt = f"""
You are a biomedical AI assistant.

Use ONLY the context below.

Context:
{context}

Question:
{query}

Rules:
- If answer is not in context say:
  "Not enough medical evidence in provided documents."
- Be precise and medically accurate.
"""

        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )

        return response.text
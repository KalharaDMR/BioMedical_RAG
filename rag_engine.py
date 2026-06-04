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

    def embed(self, text):
        return embedding_model.encode(text).astype("float32")

    # ✅ HYBRID SEARCH (BM25 + VECTOR)
    def retrieve(self, query, k=5):
        # vector search
        q_vec = self.embed(query)
        vector_results = self.vs.search(q_vec, top_k=k)

        # BM25 search
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        top_bm25_idx = np.argsort(bm25_scores)[-k:]

        bm25_results = [self.texts[i] for i in top_bm25_idx]

        # merge results
        return list(set(vector_results + bm25_results))

    def generate(self, query):
        docs = self.retrieve(query)

        context = "\n\n".join(
            [f"[DOC {i+1}] {doc}" for i, doc in enumerate(docs)]
        )

        prompt = f"""
You are a biomedical AI assistant.

Use ONLY the context below.

You MUST cite sources like [DOC 1].

If answer is not in context say:
"Not enough medical evidence in provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

        try:
            response = client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"LLM error: {str(e)}"
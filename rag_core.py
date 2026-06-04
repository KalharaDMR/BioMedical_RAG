from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from google import genai
from dotenv import load_dotenv
from datasets import load_dataset
import os
import re

# ----------------------------
# ENV
# ----------------------------
load_dotenv()

# ----------------------------
# MODELS
# ----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ----------------------------
# LOAD DATASET
# ----------------------------
print("📥 Loading dataset...")

dataset = load_dataset(
    "keivalya/MedQuad-MedicalQnADataset",
    split="train"
).select(range(800))

# ----------------------------
# BUILD DOCUMENTS
# ----------------------------
documents = []

for item in dataset:
    q = item.get("Question", "")
    a = item.get("Answer", "")

    if q and a and len(a) > 40:
        doc = f"Q: {q}\nA: {a}"
        documents.append(doc)

print(f"📄 Documents: {len(documents)}")

# ----------------------------
# TOKENIZE FOR BM25
# ----------------------------
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# ----------------------------
# EMBEDDINGS (FAISS)
# ----------------------------
print("🔄 Creating embeddings...")

embeddings = embedding_model.encode(documents, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print("✅ RAG READY (STEP 9)")

# ----------------------------
# HYBRID RETRIEVAL
# ----------------------------
def hybrid_retrieve(query, k=5):
    query_tokens = query.lower().split()

    # BM25 scores
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top = np.argsort(bm25_scores)[-k:]

    # FAISS scores
    q_emb = embedding_model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)

    _, faiss_top = index.search(q_emb, k)

    # merge results
    candidates = set(bm25_top.tolist() + faiss_top[0].tolist())

    return [documents[i] for i in candidates if i < len(documents)]

# ----------------------------
# SIMPLE RERANKER (LIGHTWEIGHT)
# ----------------------------
def rerank(query, docs):
    scored = []

    for d in docs:
        score = len(set(query.lower().split()) & set(d.lower().split()))
        scored.append((score, d))

    scored.sort(reverse=True)
    return [d for _, d in scored[:5]]

# ----------------------------
# LLM CALL (SAFE)
# ----------------------------
def call_llm(prompt):
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"LLM error: {str(e)}"

# ----------------------------
# RAG ENGINE (STEP 9)
# ----------------------------
def ask_rag(query):

    retrieved = hybrid_retrieve(query, k=8)
    reranked = rerank(query, retrieved)

    context = "\n\n".join(
        [f"[DOC {i+1}] {doc}" for i, doc in enumerate(reranked)]
    )

    print("\n🔎 TOP CONTEXT:")
    for d in reranked[:3]:
        print("-", d[:120], "...")

    prompt = f"""
You are a biomedical assistant.

Use ONLY the context below.

You MUST cite sources like [DOC 1], [DOC 2].

If answer is not in context, say:
"Not enough medical evidence in provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    return call_llm(prompt)

# ----------------------------
# CHAT LOOP
# ----------------------------
while True:
    q = input("\nAsk biomedical question (exit): ")

    if q.lower() == "exit":
        break

    ans = ask_rag(q)

    print("\n🧠 ANSWER:\n", ans)
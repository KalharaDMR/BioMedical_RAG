from fastapi import FastAPI
from ingest import load_data
from vector_store import VectorStore
from rag_engine import RAGEngine
import uvicorn

app = FastAPI()

print("📥 Loading data...")
embeddings, texts, bm25 = load_data()

print("🧠 Initializing vector DB...")
vs = VectorStore(vector_size=384)
vs.add_documents(embeddings, texts)

rag = RAGEngine(vs, texts, bm25)

@app.get("/")
def home():
    return {"status": "Biomedical RAG API running"}

@app.get("/chat")
def chat(q: str):
    answer = rag.generate(q)
    return {
        "question": q,
        "answer": answer
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
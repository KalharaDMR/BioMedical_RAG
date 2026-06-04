from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ingest import load_data
from vector_store import VectorStore
from rag_engine import RAGEngine
import uvicorn

app = FastAPI()

# ✅ FIX: middleware AFTER app creation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# request model
class ChatRequest(BaseModel):
    message: str

print("📥 Loading data...")
embeddings, texts, bm25 = load_data()

print("🧠 Initializing vector DB...")
vs = VectorStore(vector_size=384)
vs.add_documents(embeddings, texts)

rag = RAGEngine(vs, texts, bm25)

@app.get("/")
def home():
    return {"status": "Biomedical RAG API running"}

# ✅ FIX: POST instead of GET
@app.post("/chat")
def chat(req: ChatRequest):
    answer = rag.generate(req.message)
    return {
        "question": req.message,
        "answer": answer
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
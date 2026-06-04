# 🧠 Biomedical RAG System (AI-Powered Medical Q&A)

A full-stack AI application that answers biomedical questions using a **Retrieval-Augmented Generation (RAG)** pipeline with **FastAPI backend** and **React ChatGPT-style UI**.

---

## 🚀 Features

- 🔍 Hybrid Retrieval (BM25 + Vector Search)
- 🧠 Semantic Search using Sentence Transformers
- ⚡ Fast vector database (FAISS / Qdrant)
- 🤖 LLM-powered answers (Google Gemini API)
- 💬 ChatGPT-style React UI
- 🔄 Real-time query-response pipeline
- 🧾 Source-grounded answers (context-based)

---

## 🏗 System Architecture


User (React UI)
↓
FastAPI Backend
↓
RAG Engine
↓
Hybrid Retrieval
(BM25 + Vector Search)
↓
Top Relevant Documents
↓
LLM (Gemini)
↓
Final Answer


---

## 🔁 System Flow (Step-by-Step)

1. User enters a biomedical question via React UI
2. Request sent to FastAPI `/chat` endpoint
3. RAG Engine processes query:
   - Embeds query using SentenceTransformer
   - Performs:
     - Vector Search (FAISS / Qdrant)
     - BM25 Keyword Search
   - Combines results (Hybrid Retrieval)
4. Top documents are reranked
5. Context is passed to LLM (Gemini)
6. LLM generates final answer
7. Response sent back to UI

---

## 🧰 Tech Stack

### 🔹 Backend
- Python
- FastAPI
- SentenceTransformers (`all-MiniLM-L6-v2`)
- FAISS / Qdrant (Vector DB)
- BM25 (`rank_bm25`)
- Google Gemini API

### 🔹 Frontend
- React.js
- Axios
- CSS (custom Chat UI)

### 🔹 Data
- MedQuad Biomedical Dataset (HuggingFace)

---

## 📁 Project Structure


biomedical-rag/
│
├── backend/
│ ├── main.py
│ ├── rag_engine.py
│ ├── vector_store.py
│ ├── ingest.py
│
├── frontend/
│ ├── src/
│ │ ├── App.js
│ │ ├── App.css
│
├── .gitignore
├── README.md
└── requirements.txt


---

## ⚙️ Installation

### 1️⃣ Clone repository

git clone https://github.com/YOUR_USERNAME/biomedical-rag.git
cd biomedical-rag


---

### 2️⃣ Setup backend

cd backend
python -m venv venv
venv\Scripts\activate # Windows

pip install -r requirements.txt


---

### 3️⃣ Add environment variables

Create `.env` file:


GEMINI_API_KEY=your_api_key_here


---

### 4️⃣ Run backend

python main.py


Server runs at:

http://localhost:8000


---

### 5️⃣ Setup frontend

cd frontend
npm install
npm start


Frontend runs at:

http://localhost:3000


---

## 📡 API Endpoints

### GET /
Check API status

### POST /chat

Request:
```json
{
  "message": "What causes diabetes?"
}

Response:

{
  "answer": "Diabetes is caused by..."
}
🧠 Example Query

Input:


What are symptoms of anemia?


Output:


Fatigue, weakness, pale skin...

⚠️ Known Limitations
Dataset coverage is limited (MedQuad)
LLM API rate limits (Gemini free tier)
Not a medical diagnostic tool
🚀 Future Improvements
Streaming responses (real-time typing)
Conversation memory
Better reranking (cross-encoder)
Medical document upload (PDF RAG)
Deployment (AWS / Docker)
📜 Disclaimer

This system is for educational purposes only.
It should NOT be used for real medical diagnosis or treatment.

👨‍💻 Author
Ramesh Kalhara
Developed as part of a Biomedical AI / RAG system project.

⭐ Support

If you like this project:

⭐ Star the repo
🍴 Fork it
🧠 Improve it

---

# 💡 Small Improvements (optional but powerful)

You can also add:

### 1. Screenshots section
After you run UI:

```markdown
## 📸 UI Preview

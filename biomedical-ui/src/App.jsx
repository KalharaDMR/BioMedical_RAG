import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = { role: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setInput("");

    try {
      const res = await axios.post("http://localhost:8000/chat", {
        message: input,
      });
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: res.data.answer },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "Error connecting to backend." },
      ]);
    }

    setLoading(false);
  };

  return (
    <div className="app-shell">

      {/* ── Header ── */}
      <header className="chat-header">
        <div className="header-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" width="20" height="20" aria-hidden="true">
            <path d="M12 3c-1.5 3-3 4.5-3 7.5a3 3 0 0 0 6 0C15 7.5 13.5 6 12 3z"/>
            <path d="M6 12H3m18 0h-3M12 21v-3"/>
            <circle cx="12" cy="10.5" r="1"/>
          </svg>
        </div>
        <div className="header-text">
          <h1>BioMed Assistant</h1>
          <p>AI-powered biomedical Q&amp;A</p>
        </div>
        <div className="status-badge">
          <span className="status-dot" />
          <span>Online</span>
        </div>
      </header>

      {/* ── Chat area ── */}
      <div className="chat-area">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="32" height="32" aria-hidden="true">
                <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2v-4M9 21H5a2 2 0 0 1-2-2v-4m0 0h18"/>
              </svg>
            </div>
            <p>Ask any biomedical question.<br />I'm here to help with research, anatomy,<br />pharmacology, and more.</p>
          </div>
        )}

        {messages.length > 0 && (
          <div className="msg-row bot">
            <div className="avatar bot">AI</div>
            <div className="bubble bot">
              Hello! I'm your biomedical AI assistant. Ask me anything about biology, medicine, pharmacology, or clinical research.
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`msg-row ${msg.role}`}>
            <div className={`avatar ${msg.role}`}>
              {msg.role === "bot" ? "AI" : "You"}
            </div>
            <div className={`bubble ${msg.role}`}>{msg.text}</div>
          </div>
        ))}

        {loading && (
          <div className="msg-row bot">
            <div className="avatar bot">AI</div>
            <div className="typing-bubble">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
            </div>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* ── Input area ── */}
      <div className="input-area">
        <div className="input-row">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            placeholder="Ask a biomedical question…"
            disabled={loading}
            aria-label="Message input"
          />
          <button
            onClick={sendMessage}
            disabled={loading}
            className="send-btn"
            aria-label="Send message"
          >
            ↑
          </button>
        </div>
        <p className="hint">Responses are AI-generated. Always verify with clinical sources.</p>
      </div>

    </div>
  );
}

export default App;
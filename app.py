from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

print("Loading model...")

# =========================
# MODEL
# =========================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

question_embeddings = joblib.load("model/question_embeddings.pkl")
questions = joblib.load("model/questions.pkl")
answers = joblib.load("model/answers.pkl")

print("Model loaded successfully")


# =========================
# REQUEST BODY
# =========================
class Query(BaseModel):
    text: str


# =========================
# HOME PAGE (WORKING UI)
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Legal AI</title>

<style>
body {
    margin: 0;
    font-family: Arial;
    background: #0f0f0f;
    color: white;
}

.container {
    display: flex;
    height: 100vh;
}

.left {
    width: 30%;
    padding: 20px;
    background: #111;
}

.right {
    width: 70%;
    padding: 20px;
    display: flex;
    flex-direction: column;
}

input {
    width: 90%;
    padding: 10px;
    margin-top: 10px;
}

button {
    padding: 10px;
    margin-top: 10px;
    background: gold;
    border: none;
    cursor: pointer;
}

.chat {
    flex: 1;
    overflow-y: auto;
    background: #1a1a1a;
    padding: 10px;
    border-radius: 10px;
}

.msg {
    padding: 10px;
    margin: 10px;
    border-radius: 8px;
}

.user {
    background: #2563eb;
    text-align: right;
}

.bot {
    background: #222;
}

</style>
</head>

<body>

<div class="container">

<div class="left">
    <h2>⚖️ Legal AI Assistant</h2>
    <input id="text" placeholder="Ask: What is bail?">
    <button onclick="ask()">Search</button>
</div>

<div class="right">
    <h3>Chat</h3>
    <div class="chat" id="chat"></div>
</div>

</div>

<script>

async function ask() {
    let text = document.getElementById("text").value;
    let chat = document.getElementById("chat");

    if (!text) return;

    chat.innerHTML += `<div class='msg user'>${text}</div>`;

    let res = await fetch("/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text})
    });

    let data = await res.json();

    console.log(data); // DEBUG

    let answer = data.chatbot_answer || "No response from server";

    chat.innerHTML += `<div class='msg bot'>${answer}</div>`;

    document.getElementById("text").value = "";
}

</script>

</body>
</html>
"""


# =========================
# AI LOGIC
# =========================
@app.post("/ask")
def ask(query: Query):

    try:
        user_embedding = model.encode([query.text])

        scores = cosine_similarity(user_embedding, question_embeddings)[0]

        top_idx = np.argsort(scores)[::-1][:3]

        results = []
        context = []

        best_score = scores[top_idx[0]]

        # IF NO GOOD MATCH → fallback response
        if best_score < 0.35:
            return {
                "chatbot_answer": "⚖️ Sorry, I could not find a strong legal match for your question. Try rephrasing."
            }

        for i in top_idx:
            results.append({
                "question": questions[i],
                "answer": answers[i],
                "score": float(scores[i])
            })

            context.append(f"Q: {questions[i]}\nA: {answers[i]}")

        answer = "🧠 LEGAL AI RESPONSE\n\n" + "\n\n".join(context)

        return {
            "chatbot_answer": answer,
            "results": results
        }

    except Exception as e:
        return {
            "chatbot_answer": f"⚠️ Error: {str(e)}"
        }
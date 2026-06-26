# ⚖️ Legal AI Assistant

A semantic search-based Legal AI Assistant built using **Python**, **FastAPI**, and **Sentence Transformers**. The application allows users to ask legal questions and retrieves the most relevant answers from a legal question-answer dataset using AI-powered semantic similarity.

---

## 📌 Project Overview

Legal information can be difficult to search using traditional keyword matching. This project uses **Natural Language Processing (NLP)** to understand the meaning of a user's question and retrieve the most relevant legal answers.

Instead of relying only on exact keywords, the assistant compares the semantic meaning of the user's query with thousands of legal questions using sentence embeddings.

---

## ✨ Features

* 🔍 AI-powered semantic search
* ⚖️ Legal question answering
* 🚀 FastAPI backend
* 💻 Modern web interface
* 📊 Top 3 most relevant legal answers
* 🤖 Sentence Transformer embeddings
* ⚡ Fast similarity search using cosine similarity

---

## 🛠️ Tech Stack

| Technology            | Purpose              |
| --------------------- | -------------------- |
| Python                | Backend Development  |
| FastAPI               | REST API             |
| Sentence Transformers | Text Embeddings      |
| Scikit-learn          | Cosine Similarity    |
| NumPy                 | Numerical Operations |
| Joblib                | Model Storage        |
| HTML/CSS/JavaScript   | Frontend             |

---

## 📂 Project Structure

```text
legal-project/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── data/
│   └── IndicLegalQA Dataset_10K.json
└── model/
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/AmritavarshiniS/Legal-AI-Assistant.git
```

Go into the project folder:

```bash
cd Legal-AI-Assistant
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Train the embeddings:

```bash
python train.py
```

Run the application:

```bash
python -m uvicorn app:app --reload
```

Open your browser:

```text
http://127.0.0.1:8000
```

---

## 🧠 How It Works

1. User enters a legal question.
2. The question is converted into a sentence embedding.
3. Cosine similarity compares the query with the stored legal dataset.
4. The three most relevant legal answers are returned.

---

## 📸 Screenshots

> Screenshots of the application will be added here.

---

## 🔮 Future Improvements

* AI-generated conversational responses
* Login and user authentication
* Saved chat history
* Voice input
* PDF legal document analysis
* Case law search
* Deployment with a public URL
* Responsive mobile interface

---

## 👩‍💻 Author

**Amritavarshini S**

B.Tech Computer Science Engineering (Core)

Passionate about Artificial Intelligence, Machine Learning, Robotics, and Ocean Engineering.

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

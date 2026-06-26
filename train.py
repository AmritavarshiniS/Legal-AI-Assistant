import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

# Load JSON dataset
df = pd.read_json("data/IndicLegalQA Dataset_10K.json")

print("COLUMNS FOUND:", df.columns)

# Fix column names (safe fallback)
question_col = "question"
answer_col = "answer"

questions = df[question_col].astype(str).tolist()
answers = df[answer_col].astype(str).tolist()

print("\nLoading model...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Encoding questions...")

embeddings = model.encode(questions, show_progress_bar=True)

# Save
joblib.dump(model, "model/bert_model.pkl")
joblib.dump(embeddings, "model/question_embeddings.pkl")
joblib.dump(questions, "model/questions.pkl")
joblib.dump(answers, "model/answers.pkl")

print("\n✅ Training complete!")
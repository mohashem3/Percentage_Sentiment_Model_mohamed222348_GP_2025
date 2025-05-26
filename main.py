from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from transformers import pipeline

# ===== Download required NLTK resources =====
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')  # Optional

# ===== Load model components =====
model = joblib.load("model/svm_model.joblib")
vectorizer = joblib.load("model/tfidf_vectorizer.joblib")
ratios = joblib.load("model/nb_ratios.joblib")

# ===== FastAPI setup =====
app = FastAPI()

# ===== CORS setup for Vue frontend =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Preprocessing setup =====
stop_words_set = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def handle_negations(text):
    return re.sub(r"\b(not|n't)\s+(\w+)", r"not_\2", text)

def clean_text(text):
    text = text.lower()
    text = handle_negations(text)
    text = re.sub('<br />', '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    tokens = word_tokenize(text)
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words_set]
    return " ".join(filtered)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ===== Prediction Logic =====
def predict_sentiment(text):
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    tfidf_nb = tfidf.multiply(ratios)
    raw_score = model.decision_function(tfidf_nb)[0]
    prob = sigmoid(raw_score)
    label = "positive" if raw_score >= 0 else "negative"
    percentage = round(prob * 100 if label == "positive" else (1 - prob) * 100)
    return {"sentiment": label, "confidence": percentage}

# ===== Request schema & endpoint =====
class ReviewInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: ReviewInput):
    result = predict_sentiment(data.text)
    return result

# ===== Summarization Setup =====
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class ReviewList(BaseModel):
    reviews: List[str]

@app.post("/summarize")
def summarize_reviews(data: ReviewList):
    reviews = data.reviews

    if not reviews:
        return {"summary": "No reviews provided."}

    combined_text = " ".join(reviews)

    if len(combined_text.split()) < 30:
        return {"summary": combined_text}  # Not enough content for summarization

    result = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)
    return {"summary": result[0]["summary_text"]}

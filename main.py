from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load components
model = joblib.load("models/svm_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
ratios = joblib.load("models/nb_ratios.joblib")

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

def predict_sentiment(text):
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    tfidf_nb = tfidf.multiply(ratios)
    raw_score = model.decision_function(tfidf_nb)[0]
    prob = sigmoid(raw_score)
    label = "positive" if raw_score >= 0 else "negative"
    percentage = round(prob * 100 if label == "positive" else (1 - prob) * 100)
    return {"sentiment": label, "confidence": percentage}

# FastAPI setup
app = FastAPI()

class ReviewInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: ReviewInput):
    result = predict_sentiment(data.text)
    return result

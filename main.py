from fastapi import FastAPI, HTTPException
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
import os
from dotenv import load_dotenv

# New OpenAI SDK import style
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

# ===== GPT-3.5 Summarization Setup =====
class ReviewList(BaseModel):
    reviews: List[str]

@app.post("/summarize")
def summarize_reviews(data: ReviewList):
    try:
        reviews = data.reviews
        if not reviews:
            return {"summary": "No reviews provided."}
        combined_text = " ".join(reviews)
        if len(combined_text.split()) < 30:
            return {
                "summary": combined_text,
                "tone": "undetermined",
                "positives": [],
                "negatives": []
            }

        # Use GPT to return structured summary
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that summarizes user-submitted movie reviews into a paragraph and extracts key insights."
                },
                {
                    "role": "user",
                    "content": f"""
The following are user-submitted reviews for a movie:

{combined_text}

Please do the following:
1. Write a natural, informative summary paragraph that reflects the reviews.
2. Clearly state the overall tone (positive, negative, or mixed).
3. List the main positive points users mentioned.
4. List the common criticisms or negative points.

Respond in this exact JSON format:
{{
  "summary": "<your paragraph summary>",
  "tone": "<overall tone>",
  "positives": ["<positive point 1>", "<positive point 2>", "..."],
  "negatives": ["<negative point 1>", "<negative point 2>", "..."]
}}
"""
                }
            ],
            max_tokens=300,
            temperature=0.5,
        )

        response_text = response.choices[0].message.content.strip()

        import json
        parsed = json.loads(response_text)
        return parsed

    except Exception as e:
        print(f"Error in /summarize: {e}")
        raise HTTPException(status_code=500, detail="Error generating summary.")

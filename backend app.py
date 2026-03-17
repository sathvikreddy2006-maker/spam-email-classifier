from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

# Load your pre-trained model and vectorizer
# Ensure you've saved them using joblib.dump()
model = joblib.load("model/classifier.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

class EmailRequest(BaseModel):
    content: str

@app.post("/predict")
async def predict_email(request: EmailRequest):
    # 1. Transform input text
    data = vectorizer.transform([request.content])
    # 2. Predict
    prediction = model.predict(data)[0]
    # 3. Return result (e.g., 1 for Fake/Spam, 0 for Real/Ham)
    label = "Fake/Spam" if prediction == 1 else "Real/Legit"
    return {"prediction": label}

# Serve the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schemas import ReviewRequest, ReviewResponse
from agent_core import predict, MODEL_ID

APP_NAME = os.getenv("APP_NAME", "review-agent")
app = FastAPI(title=APP_NAME)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/predict", response_model=ReviewResponse)
def predict_endpoint(payload: ReviewRequest):
    result = predict(payload.review)
    return JSONResponse(result)

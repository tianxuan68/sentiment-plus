from fastapi import APIRouter

from app.schemas.predict import PredictRequest, PredictResponse
from app.services.bert_sentiment import predict_sentiment

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/api/sentiment/predict", response_model=PredictResponse)
def sentiment_predict(body: PredictRequest):
    """整句情感分类推理入口（交 1 号接入）。"""
    results = predict_sentiment(body.texts)
    return PredictResponse(results=results)

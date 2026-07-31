"""
推理路由：对接 1 号（郑平高）
不改路径与响应字段
"""

# 1.导包
from fastapi import APIRouter

from app.schemas.predict import PredictRequest, PredictResponse
from app.services.bert_sentiment import predict_sentiment

router = APIRouter()


@router.get('/health')
def health():
    # 健康检查
    return {'status': 'ok'}


@router.post('/api/sentiment/predict', response_model=PredictResponse)
def sentiment_predict(body: PredictRequest):
    # 整句情感分类推理入口
    results = predict_sentiment(body.texts)
    return PredictResponse(results=results)

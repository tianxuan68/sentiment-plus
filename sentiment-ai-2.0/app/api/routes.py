from fastapi import APIRouter

from app.schemas.tags import (
    DynamicTagPredictRequest,
    TagPredictRequest,
    TagPredictResponse,
)
from app.services.tag_predict import predict_dynamic_tags, predict_fixed_tags

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "service": "sentiment-ai-2.0", "version": "2.0.0-dev"}


@router.post("/api/v2/tags/predict", response_model=TagPredictResponse)
def tags_predict(body: TagPredictRequest):
    """2.0-A：固定标签集多标签预测（当前为占位实现）。"""
    return predict_fixed_tags(body)


@router.post("/api/v2/tags/predict_dynamic", response_model=TagPredictResponse)
def tags_predict_dynamic(body: DynamicTagPredictRequest):
    """2.0-B：按商品/类目动态标签（当前为占位，等安排再实现）。"""
    return predict_dynamic_tags(body)

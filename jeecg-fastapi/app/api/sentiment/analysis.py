from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.deps import get_current_user
from app.models.entities import SysUser
from app.schemas.response import Result
from app.schemas.sentiment.bodies import (
    SentimentBatchBody,
    SentimentKeywordsBody,
    SentimentPredictBody,
)
from app.services import sentiment_service

router = APIRouter(prefix="/sentimentAnalysis", tags=["评论情感分析"])


@router.get("/overview")
def sentiment_overview(user: SysUser = Depends(get_current_user)):
    return Result.ok(sentiment_service.overview_stats())


@router.post("/predict")
def sentiment_predict(body: SentimentPredictBody, user: SysUser = Depends(get_current_user)):
    try:
        return Result.ok(sentiment_service.predict(body.text, model=body.model or "baseline"))
    except ValueError as exc:
        return Result.error(str(exc))


@router.post("/predictBatch")
def sentiment_predict_batch(body: SentimentBatchBody, user: SysUser = Depends(get_current_user)):
    try:
        return Result.ok(sentiment_service.predict_batch(body.texts, model=body.model or "baseline"))
    except ValueError as exc:
        return Result.error(str(exc))


@router.post("/keywords")
def sentiment_keywords(body: SentimentKeywordsBody, user: SysUser = Depends(get_current_user)):
    try:
        return Result.ok(sentiment_service.keywords(body.text, top_n=body.topN))
    except ValueError as exc:
        return Result.error(str(exc))


@router.post("/aspects")
def sentiment_aspects(body: SentimentPredictBody, user: SysUser = Depends(get_current_user)):
    try:
        return Result.ok(sentiment_service.aspects(body.text))
    except ValueError as exc:
        return Result.error(str(exc))

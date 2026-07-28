from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="评论文本")
    top_n: int = Field(5, ge=1, le=20, description="关键词 Top-N")


class BatchAnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=10, description="批量评论，最多 10 条")
    top_n: int = Field(5, ge=1, le=20, description="关键词 Top-N")


class SentimentResult(BaseModel):
    label: int
    label_text: str
    prob: float


class KeywordItem(BaseModel):
    word: str
    score: float


class AspectItem(BaseModel):
    aspect: str
    polarity: int
    polarity_text: str
    prob: float


class AspectEntity(BaseModel):
    """属性 NER 实体片段，供前端高亮展示。"""
    aspect: str
    text: str
    start: int
    end: int
    polarity: int
    polarity_text: str


class AnalyzeResult(BaseModel):
    sentiment: SentimentResult
    keywords: list[KeywordItem]
    aspects: list[AspectItem]
    entities: list[AspectEntity] = Field(default_factory=list)
    active_model: str = Field("BERT", description="当前主推理模型")
    source: str = "mock"
    latency_ms: Optional[float] = None
    text_preview: Optional[str] = None


class BatchAnalyzeResult(BaseModel):
    items: list[AnalyzeResult]
    total: int
    avg_latency_ms: Optional[float] = None


class PolarityStat(BaseModel):
    positive: int
    negative: int
    total: int
    positive_ratio: float
    negative_ratio: float


class TrendPoint(BaseModel):
    name: str
    value: int
    positive: Optional[int] = None
    negative: Optional[int] = None


class ProsConsItem(BaseModel):
    word: str
    score: float


class ProsConsResult(BaseModel):
    pros: list[ProsConsItem]
    cons: list[ProsConsItem]
    source: str = "mock"


class DashboardSummary(BaseModel):
    """评价看板 KPI；统计岗交付前由 Mock 稳定演示。"""
    satisfaction_score: float = Field(..., description="综合满意度 0-100")
    avg_rating: float = Field(..., description="平均评分 1-5")
    review_count: int
    positive_ratio: float
    trend_direction: Literal["up", "down", "stable"] = "up"
    source: str = "mock"


class AiStatusResult(BaseModel):
    """sentiment-ai 联调状态；前端展示 Mock / 在线 / 降级。"""
    mock_enabled: bool
    ai_base_url: str
    ai_reachable: bool
    ai_latency_ms: Optional[float] = None
    mode: Literal["mock", "live", "degraded"] = "mock"


class ModelMetric(BaseModel):
    model: str
    acc: float
    f1: float
    owner: str = ""


class ModelCompareResult(BaseModel):
    """三模型情感对比链：Baseline → BiLSTM → BERT。"""
    metrics: list[ModelMetric]
    best_model: str
    bilstm_vs_baseline_f1_gain: float = Field(..., description="BiLSTM 相对 Baseline 的 F1 提升")
    source: str = "mock"


class KeywordCompareItem(BaseModel):
    word: str
    keybert_score: float
    tfidf_score: float


class KeywordCompareResult(BaseModel):
    keybert: list[KeywordItem]
    tfidf: list[KeywordItem]
    overlap: list[KeywordCompareItem]
    keybert_unique_count: int
    source: str = "mock"


class RatingBucket(BaseModel):
    rating: int
    count: int
    ratio: float


class RatingDistribution(BaseModel):
    buckets: list[RatingBucket]
    avg_rating: float
    total: int
    source: str = "mock"

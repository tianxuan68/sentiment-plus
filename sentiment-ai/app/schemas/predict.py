"""
请求/响应 Schema（对接 1 号，字段勿改）
"""

# 1.导包
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description='待预测评论文本列表')


class PredictItem(BaseModel):
    text: str
    label: int
    label_name: str
    confidence: float


class PredictResponse(BaseModel):
    results: list[PredictItem]

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SentimentPredictBody(BaseModel):
    text: str = Field(..., min_length=1)
    model: Optional[str] = Field("baseline")


class SentimentBatchBody(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    model: Optional[str] = Field("baseline")


class SentimentKeywordsBody(BaseModel):
    text: str = Field(..., min_length=1)
    topN: int = Field(10, ge=1, le=50)

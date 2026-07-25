from __future__ import annotations

from typing import Any

from app.services.sentiment import (
    analyze_aspects,
    extract_keywords,
    get_demo_stats,
    predict_sentiment,
    predict_sentiment_batch,
)


def predict(text: str, model: str = "baseline") -> dict[str, Any]:
    return predict_sentiment(text, model=model)


def predict_batch(texts: list[str], model: str = "baseline") -> dict[str, Any]:
    return predict_sentiment_batch(texts, model=model)


def keywords(text: str, top_n: int = 10) -> dict[str, Any]:
    return extract_keywords(text, top_n=top_n)


def aspects(text: str) -> dict[str, Any]:
    return analyze_aspects(text)


def overview_stats() -> dict[str, Any]:
    return get_demo_stats()

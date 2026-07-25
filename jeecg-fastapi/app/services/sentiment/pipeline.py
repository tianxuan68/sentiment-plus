from __future__ import annotations

import re
from typing import Any

from app.services.sentiment.lexicon import ASPECT_LEXICON, NEGATIVE_WORDS, POSITIVE_WORDS

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}")


def _score_text(text: str) -> tuple[float, list[str], list[str]]:
    pos_hits = [w for w in POSITIVE_WORDS if w in text]
    neg_hits = [w for w in NEGATIVE_WORDS if w in text]
    raw = len(pos_hits) - len(neg_hits)
    score = max(-1.0, min(1.0, raw / 3.0))
    return score, pos_hits, neg_hits


def predict_sentiment(text: str, model: str = "baseline") -> dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("评论文本不能为空")
    used_model = model if model == "baseline" else "baseline"
    score, pos_hits, neg_hits = _score_text(cleaned)
    if score > 0.15:
        label, label_text = 1, "正向"
    elif score < -0.15:
        label, label_text = 0, "负向"
    else:
        label, label_text = -1, "中性"
    return {
        "text": cleaned,
        "label": label,
        "labelText": label_text,
        "score": round(score, 4),
        "confidence": round(min(0.95, 0.55 + abs(score) * 0.4), 4),
        "model": used_model,
        "requestedModel": model,
        "positiveHits": pos_hits[:8],
        "negativeHits": neg_hits[:8],
        "note": None if model == "baseline" else "深度模型尚未部署，已使用 baseline 推理",
    }


def predict_sentiment_batch(texts: list[str], model: str = "baseline") -> dict[str, Any]:
    items = [predict_sentiment(t, model=model) for t in texts if (t or "").strip()]
    return {
        "total": len(items),
        "positive": sum(1 for i in items if i["label"] == 1),
        "negative": sum(1 for i in items if i["label"] == 0),
        "neutral": sum(1 for i in items if i["label"] == -1),
        "items": items,
    }


def extract_keywords(text: str, top_n: int = 10) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("评论文本不能为空")
    tokens = _TOKEN_RE.findall(cleaned)
    freq: dict[str, int] = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0) + 1
    scored: list[tuple[str, float]] = []
    for tok, cnt in freq.items():
        boost = 1.0
        if any(w in tok or tok in w for w in POSITIVE_WORDS):
            boost += 0.8
        if any(w in tok or tok in w for w in NEGATIVE_WORDS):
            boost += 0.8
        scored.append((tok, cnt * boost))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return {
        "text": cleaned,
        "keywords": [{"word": w, "score": round(s, 4)} for w, s in scored[:top_n]],
        "method": "frequency+lexicon",
    }


def analyze_aspects(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("评论文本不能为空")
    aspects: list[dict[str, Any]] = []
    pred = predict_sentiment(cleaned)
    for aspect, triggers in ASPECT_LEXICON.items():
        hits = [t for t in triggers if t in cleaned]
        if not hits:
            continue
        aspects.append(
            {
                "aspect": aspect,
                "polarity": pred["labelText"],
                "label": pred["label"],
                "score": pred["score"],
                "triggers": hits,
            }
        )
    return {"text": cleaned, "aspects": aspects, "method": "lexicon-ner"}


def get_demo_stats() -> dict[str, Any]:
    return {
        "totalReviews": 45210,
        "positiveRate": 0.628,
        "negativeRate": 0.291,
        "neutralRate": 0.081,
        "modelAccuracy": {"baseline": 0.86, "cnn": 0.89, "bilstm": 0.91, "bert": 0.93},
        "prosTop10": [
            {"word": "物流快", "score": 0.92},
            {"word": "质量好", "score": 0.90},
            {"word": "实惠", "score": 0.88},
            {"word": "包装完好", "score": 0.85},
            {"word": "好用", "score": 0.84},
            {"word": "客服耐心", "score": 0.82},
            {"word": "性价比高", "score": 0.81},
            {"word": "味道不错", "score": 0.79},
            {"word": "做工精细", "score": 0.77},
            {"word": "值得回购", "score": 0.76},
        ],
        "consTop10": [
            {"word": "发货慢", "score": 0.91},
            {"word": "有异味", "score": 0.88},
            {"word": "偏贵", "score": 0.86},
            {"word": "易破损", "score": 0.84},
            {"word": "客服敷衍", "score": 0.82},
            {"word": "描述不符", "score": 0.80},
            {"word": "掉色", "score": 0.78},
            {"word": "尺码不准", "score": 0.76},
            {"word": "味道一般", "score": 0.74},
            {"word": "难清洗", "score": 0.72},
        ],
        "trend": [
            {"month": "2025-10", "positive": 0.61, "negative": 0.30},
            {"month": "2025-11", "positive": 0.63, "negative": 0.28},
            {"month": "2025-12", "positive": 0.60, "negative": 0.31},
            {"month": "2026-01", "positive": 0.64, "negative": 0.27},
            {"month": "2026-02", "positive": 0.65, "negative": 0.26},
            {"month": "2026-03", "positive": 0.62, "negative": 0.29},
        ],
        "aspectCoverage": [
            {"aspect": "质量", "count": 12840},
            {"aspect": "价格", "count": 10220},
            {"aspect": "物流", "count": 9650},
            {"aspect": "服务", "count": 7340},
            {"aspect": "口味", "count": 5160},
        ],
        "dataNote": "演示统计；接入真实评论库后由服务端聚合替换",
    }

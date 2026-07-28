"""sentiment-ai 客户端：Mock 可演示，关闭 Mock 后并行转发真实推理。"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

import httpx

from app.core.config import settings
from app.schemas.review import (
    AiStatusResult,
    AnalyzeResult,
    AspectEntity,
    AspectItem,
    BatchAnalyzeResult,
    DashboardSummary,
    KeywordCompareItem,
    KeywordCompareResult,
    KeywordItem,
    ModelCompareResult,
    ModelMetric,
    PolarityStat,
    ProsConsItem,
    ProsConsResult,
    RatingBucket,
    RatingDistribution,
    SentimentResult,
    TrendPoint,
)

logger = logging.getLogger(__name__)

_ASPECT_HINTS = ("质量", "价格", "物流", "包装", "服务", "外观", "性价比", "续航")
_TEXT_PREVIEW_LEN = 48


def _label_text(label: int) -> str:
    return "正向" if label == 1 else "负向"


def _text_preview(text: str) -> str:
    compact = text.strip().replace("\n", " ")
    if len(compact) <= _TEXT_PREVIEW_LEN:
        return compact
    return compact[: _TEXT_PREVIEW_LEN - 1] + "…"


def _extract_entities(text: str, aspects: list[AspectItem]) -> list[AspectEntity]:
    entities: list[AspectEntity] = []
    seen: set[tuple[int, int]] = set()
    for item in aspects:
        idx = text.find(item.aspect)
        if idx < 0:
            continue
        span = (idx, idx + len(item.aspect))
        if span in seen:
            continue
        seen.add(span)
        entities.append(
            AspectEntity(
                aspect=item.aspect,
                text=item.aspect,
                start=idx,
                end=idx + len(item.aspect),
                polarity=item.polarity,
                polarity_text=item.polarity_text,
            )
        )
    return sorted(entities, key=lambda e: e.start)


def _mock_analyze(text: str, top_n: int) -> AnalyzeResult:
    neg_words = ("差", "烂", "慢", "贵", "简陋", "失望", "退货", "破损")
    is_neg = any(w in text for w in neg_words)
    label = 0 if is_neg else 1
    prob = 0.86 if is_neg else 0.92

    tokens = [t for t in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", text)]
    if not tokens:
        tokens = ["商品", "体验", "服务"]
    keywords = [
        KeywordItem(word=w, score=round(0.9 - i * 0.08, 2))
        for i, w in enumerate(tokens[:top_n])
    ]

    aspects: list[AspectItem] = []
    for aspect in _ASPECT_HINTS:
        if aspect in text:
            pol = 0 if is_neg and aspect in ("包装", "价格", "物流") else (0 if is_neg else 1)
            if "很快" in text and aspect == "物流":
                pol = 1
            if "简陋" in text and aspect == "包装":
                pol = 0
            aspects.append(
                AspectItem(
                    aspect=aspect,
                    polarity=pol,
                    polarity_text=_label_text(pol),
                    prob=0.88 if pol == 1 else 0.79,
                )
            )
    if not aspects:
        aspects = [
            AspectItem(aspect="整体", polarity=label, polarity_text=_label_text(label), prob=prob),
        ]

    return AnalyzeResult(
        sentiment=SentimentResult(label=label, label_text=_label_text(label), prob=prob),
        keywords=keywords,
        aspects=aspects,
        entities=_extract_entities(text, aspects),
        active_model="BERT",
        source="mock",
        text_preview=_text_preview(text),
    )


def _mock_polarity() -> PolarityStat:
    positive, negative = 12680, 4320
    total = positive + negative
    return PolarityStat(
        positive=positive,
        negative=negative,
        total=total,
        positive_ratio=round(positive / total, 4),
        negative_ratio=round(negative / total, 4),
    )


def _mock_trend() -> list[TrendPoint]:
    days = [
        ("03-01", 420, 310, 110),
        ("03-02", 455, 340, 115),
        ("03-03", 488, 360, 128),
        ("03-04", 502, 375, 127),
        ("03-05", 536, 400, 136),
        ("03-06", 510, 385, 125),
        ("03-07", 560, 420, 140),
    ]
    return [
        TrendPoint(name=d, value=total, positive=pos, negative=neg)
        for d, total, pos, neg in days
    ]


def _mock_pros_cons() -> ProsConsResult:
    return ProsConsResult(
        pros=[
            ProsConsItem(word="物流快", score=0.91),
            ProsConsItem(word="性价比高", score=0.88),
            ProsConsItem(word="质量不错", score=0.85),
            ProsConsItem(word="客服耐心", score=0.82),
            ProsConsItem(word="外观好看", score=0.80),
            ProsConsItem(word="安装简单", score=0.78),
            ProsConsItem(word="续航满意", score=0.76),
            ProsConsItem(word="包装完好", score=0.74),
            ProsConsItem(word="发货及时", score=0.72),
            ProsConsItem(word="手感舒适", score=0.70),
        ],
        cons=[
            ProsConsItem(word="包装简陋", score=0.89),
            ProsConsItem(word="说明书不清", score=0.84),
            ProsConsItem(word="偏贵", score=0.81),
            ProsConsItem(word="异味", score=0.78),
            ProsConsItem(word="配件少", score=0.75),
            ProsConsItem(word="做工粗糙", score=0.73),
            ProsConsItem(word="噪音大", score=0.71),
            ProsConsItem(word="售后慢", score=0.69),
            ProsConsItem(word="色差", score=0.67),
            ProsConsItem(word="容易掉漆", score=0.65),
        ],
        source="mock",
    )


def _mock_dashboard_summary() -> DashboardSummary:
    polarity = _mock_polarity()
    trend = _mock_trend()
    direction: str = "stable"
    if len(trend) >= 2 and trend[-1].positive and trend[-2].positive:
        delta = trend[-1].positive - trend[-2].positive
        if delta > 5:
            direction = "up"
        elif delta < -5:
            direction = "down"
    return DashboardSummary(
        satisfaction_score=round(polarity.positive_ratio * 100, 1),
        avg_rating=round(3.2 + polarity.positive_ratio * 1.6, 2),
        review_count=polarity.total,
        positive_ratio=polarity.positive_ratio,
        trend_direction=direction,  # type: ignore[arg-type]
        source="mock",
    )


def _mock_model_compare() -> ModelCompareResult:
    metrics = [
        ModelMetric(model="Baseline (TF-IDF+RF)", acc=0.872, f1=0.861, owner="毛鑫泽"),
        ModelMetric(model="BiLSTM+Attention", acc=0.903, f1=0.899, owner="陈江平"),
        ModelMetric(model="BERT Fine-tune", acc=0.934, f1=0.921, owner="胡潇潇"),
    ]
    return ModelCompareResult(
        metrics=metrics,
        best_model="BERT Fine-tune",
        bilstm_vs_baseline_f1_gain=round(metrics[1].f1 - metrics[0].f1, 3),
        source="mock",
    )


def _mock_keyword_compare(text: str, top_n: int) -> KeywordCompareResult:
    tokens = [t for t in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", text)]
    if not tokens:
        tokens = ["商品", "体验", "服务"]
    keybert = [
        KeywordItem(word=w, score=round(0.92 - i * 0.07, 2))
        for i, w in enumerate(tokens[:top_n])
    ]
    tfidf = [
        KeywordItem(word=w, score=round(0.78 - i * 0.06, 2))
        for i, w in enumerate(reversed(tokens[:top_n]))
    ]
    kb_words = {item.word for item in keybert}
    tf_words = {item.word for item in tfidf}
    overlap_words = kb_words & tf_words
    overlap = [
        KeywordCompareItem(
            word=w,
            keybert_score=next(i.score for i in keybert if i.word == w),
            tfidf_score=next(i.score for i in tfidf if i.word == w),
        )
        for w in overlap_words
    ]
    return KeywordCompareResult(
        keybert=keybert,
        tfidf=tfidf,
        overlap=overlap,
        keybert_unique_count=len(kb_words - tf_words),
        source="mock",
    )


def _mock_rating_distribution() -> RatingDistribution:
    buckets_raw = [
        (1, 2180),
        (2, 3540),
        (3, 6120),
        (4, 4890),
        (5, 4270),
    ]
    total = sum(count for _, count in buckets_raw)
    buckets = [
        RatingBucket(rating=r, count=c, ratio=round(c / total, 4))
        for r, c in buckets_raw
    ]
    weighted = sum(r * c for r, c in buckets_raw)
    return RatingDistribution(
        buckets=buckets,
        avg_rating=round(weighted / total, 2),
        total=total,
        source="mock",
    )


def _unwrap_ai(payload: dict[str, Any]) -> dict[str, Any]:
    """兼容 sentiment-ai `{code:0, result}` 与直接 result 字典。"""
    if not isinstance(payload, dict):
        return {}
    if "result" in payload:
        result = payload.get("result")
        return result if isinstance(result, dict) else {}
    return payload


async def _post_ai(client: httpx.AsyncClient, path: str, body: dict[str, Any]) -> dict[str, Any]:
    url = f"{settings.sentiment_ai_base_url.rstrip('/')}{path}"
    resp = await client.post(url, json=body)
    resp.raise_for_status()
    return _unwrap_ai(resp.json())


async def _real_analyze(text: str, top_n: int) -> AnalyzeResult:
    timeout = httpx.Timeout(settings.sentiment_ai_timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        sent_task = _post_ai(client, "/api/v1/sentiment/predict", {"text": text})
        kw_task = _post_ai(client, "/api/v1/keywords/extract", {"text": text, "top_n": top_n})
        aspect_task = _post_ai(
            client,
            "/api/v1/aspect/predict",
            {"text": text, "aspects": list(_ASPECT_HINTS)},
        )
        sent_raw, kw_raw, aspect_raw = await asyncio.gather(sent_task, kw_task, aspect_task)

    label = int(sent_raw.get("label", 1))
    prob = float(sent_raw.get("prob", 0.5))
    keywords_raw = kw_raw.get("keywords") or []
    keywords = [
        KeywordItem(
            word=str(item.get("word") or item.get("keyword") or ""),
            score=float(item.get("score", 0)),
        )
        for item in keywords_raw
        if isinstance(item, dict) and (item.get("word") or item.get("keyword"))
    ][:top_n]

    aspects_raw = aspect_raw.get("items") or aspect_raw.get("aspects") or []
    aspects: list[AspectItem] = []
    for item in aspects_raw:
        if not isinstance(item, dict):
            continue
        pol = int(item.get("polarity", item.get("label", 1)))
        aspects.append(
            AspectItem(
                aspect=str(item.get("aspect") or item.get("name") or "属性"),
                polarity=pol,
                polarity_text=str(item.get("polarity_text") or _label_text(pol)),
                prob=float(item.get("prob", 0.5)),
            )
        )

    return AnalyzeResult(
        sentiment=SentimentResult(label=label, label_text=_label_text(label), prob=prob),
        keywords=keywords or _mock_analyze(text, top_n).keywords,
        aspects=aspects or _mock_analyze(text, top_n).aspects,
        entities=_extract_entities(text, aspects) if aspects else _mock_analyze(text, top_n).entities,
        active_model="BERT",
        source="ai",
        text_preview=_text_preview(text),
    )


async def analyze_review(text: str, top_n: int = 5) -> AnalyzeResult:
    started = time.perf_counter()
    if settings.sentiment_ai_mock:
        result = _mock_analyze(text, top_n)
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return result
    try:
        result = await _real_analyze(text, top_n)
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return result
    except Exception as exc:
        logger.warning("sentiment-ai analyze failed, fallback mock: %s", exc)
        result = _mock_analyze(text, top_n)
        result.source = "mock_fallback"
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return result


async def analyze_batch(texts: list[str], top_n: int = 5) -> BatchAnalyzeResult:
    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        return BatchAnalyzeResult(items=[], total=0, avg_latency_ms=0)

    started = time.perf_counter()
    items = await asyncio.gather(*(analyze_review(text, top_n) for text in cleaned))
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    avg = round(elapsed / len(items), 1) if items else 0
    return BatchAnalyzeResult(items=list(items), total=len(items), avg_latency_ms=avg)


async def get_polarity_stats() -> PolarityStat:
    if settings.sentiment_ai_mock:
        return _mock_polarity()
    # 统计岗交付前：真实模式也先返回稳定 Mock
    return _mock_polarity()


async def get_trend_stats() -> list[TrendPoint]:
    if settings.sentiment_ai_mock:
        return _mock_trend()
    return _mock_trend()


async def get_dashboard_summary() -> DashboardSummary:
    if settings.sentiment_ai_mock:
        return _mock_dashboard_summary()
    return _mock_dashboard_summary()


async def get_pros_cons() -> ProsConsResult:
    if settings.sentiment_ai_mock:
        return _mock_pros_cons()
    try:
        timeout = httpx.Timeout(settings.sentiment_ai_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            raw = await _post_ai(client, "/api/v1/pros_cons/extract", {})
        pros = [
            ProsConsItem(word=str(i.get("word", "")), score=float(i.get("score", 0)))
            for i in (raw.get("pros") or [])
            if isinstance(i, dict)
        ]
        cons = [
            ProsConsItem(word=str(i.get("word", "")), score=float(i.get("score", 0)))
            for i in (raw.get("cons") or [])
            if isinstance(i, dict)
        ]
        if pros or cons:
            return ProsConsResult(pros=pros[:10], cons=cons[:10], source="ai")
    except Exception as exc:
        logger.warning("sentiment-ai pros_cons failed, fallback mock: %s", exc)
    result = _mock_pros_cons()
    result.source = "mock_fallback"
    return result


async def get_model_compare() -> ModelCompareResult:
    if settings.sentiment_ai_mock:
        return _mock_model_compare()
    try:
        timeout = httpx.Timeout(settings.sentiment_ai_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            raw = await _post_ai(client, "/api/v1/sentiment/models/compare", {})
        metrics_raw = raw.get("metrics") or []
        metrics = [
            ModelMetric(
                model=str(m.get("model", "")),
                acc=float(m.get("acc", 0)),
                f1=float(m.get("f1", 0)),
                owner=str(m.get("owner", "")),
            )
            for m in metrics_raw
            if isinstance(m, dict) and m.get("model")
        ]
        if metrics:
            best = str(raw.get("best_model") or metrics[-1].model)
            gain = float(raw.get("bilstm_vs_baseline_f1_gain", raw.get("bilstm_vs_cnn_f1_gain", 0)))
            return ModelCompareResult(
                metrics=metrics,
                best_model=best,
                bilstm_vs_baseline_f1_gain=gain,
                source="ai",
            )
    except Exception as exc:
        logger.warning("sentiment-ai model compare failed, fallback mock: %s", exc)
    result = _mock_model_compare()
    result.source = "mock_fallback"
    return result


async def compare_keywords(text: str, top_n: int = 8) -> KeywordCompareResult:
    cleaned = text.strip()
    if not cleaned:
        return KeywordCompareResult(
            keybert=[],
            tfidf=[],
            overlap=[],
            keybert_unique_count=0,
            source="mock",
        )
    if settings.sentiment_ai_mock:
        return _mock_keyword_compare(cleaned, top_n)
    try:
        timeout = httpx.Timeout(settings.sentiment_ai_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            raw = await _post_ai(
                client,
                "/api/v1/keywords/compare",
                {"text": cleaned, "top_n": top_n},
            )
        keybert = [
            KeywordItem(word=str(i.get("word", "")), score=float(i.get("score", 0)))
            for i in (raw.get("keybert") or [])
            if isinstance(i, dict) and i.get("word")
        ]
        tfidf = [
            KeywordItem(word=str(i.get("word", "")), score=float(i.get("score", 0)))
            for i in (raw.get("tfidf") or [])
            if isinstance(i, dict) and i.get("word")
        ]
        overlap_raw = raw.get("overlap") or []
        overlap = [
            KeywordCompareItem(
                word=str(i.get("word", "")),
                keybert_score=float(i.get("keybert_score", 0)),
                tfidf_score=float(i.get("tfidf_score", 0)),
            )
            for i in overlap_raw
            if isinstance(i, dict) and i.get("word")
        ]
        if keybert or tfidf:
            return KeywordCompareResult(
                keybert=keybert[:top_n],
                tfidf=tfidf[:top_n],
                overlap=overlap,
                keybert_unique_count=int(raw.get("keybert_unique_count", 0)),
                source="ai",
            )
    except Exception as exc:
        logger.warning("sentiment-ai keyword compare failed, fallback mock: %s", exc)
    result = _mock_keyword_compare(cleaned, top_n)
    result.source = "mock_fallback"
    return result


async def get_rating_distribution() -> RatingDistribution:
    if settings.sentiment_ai_mock:
        return _mock_rating_distribution()
    try:
        timeout = httpx.Timeout(settings.sentiment_ai_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            raw = await _post_ai(client, "/api/v1/analytics/rating_distribution", {})
        buckets_raw = raw.get("buckets") or []
        buckets = [
            RatingBucket(
                rating=int(b.get("rating", 0)),
                count=int(b.get("count", 0)),
                ratio=float(b.get("ratio", 0)),
            )
            for b in buckets_raw
            if isinstance(b, dict)
        ]
        if buckets:
            return RatingDistribution(
                buckets=buckets,
                avg_rating=float(raw.get("avg_rating", 0)),
                total=int(raw.get("total", 0)),
                source="ai",
            )
    except Exception as exc:
        logger.warning("sentiment-ai rating distribution failed, fallback mock: %s", exc)
    result = _mock_rating_distribution()
    result.source = "mock_fallback"
    return result


async def get_ai_status() -> AiStatusResult:
    mock_enabled = settings.sentiment_ai_mock
    base_url = settings.sentiment_ai_base_url.rstrip("/")
    if mock_enabled:
        return AiStatusResult(
            mock_enabled=True,
            ai_base_url=base_url,
            ai_reachable=False,
            mode="mock",
        )

    started = time.perf_counter()
    try:
        timeout = httpx.Timeout(min(settings.sentiment_ai_timeout, 2.0))
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{base_url}/health")
            resp.raise_for_status()
            payload = resp.json()
            ok = payload.get("status") == "ok" if isinstance(payload, dict) else True
            latency = round((time.perf_counter() - started) * 1000, 1)
            return AiStatusResult(
                mock_enabled=False,
                ai_base_url=base_url,
                ai_reachable=ok,
                ai_latency_ms=latency,
                mode="live" if ok else "degraded",
            )
    except Exception as exc:
        logger.warning("sentiment-ai health check failed: %s", exc)
        return AiStatusResult(
            mock_enabled=False,
            ai_base_url=base_url,
            ai_reachable=False,
            mode="degraded",
        )

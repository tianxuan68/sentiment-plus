"""BIO + polarity 双标注：读抽样池、半自动标注、导出中间结构。"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

import jieba

from pipelines.aspect_ner.constants import (
    ASPECT_KEYWORDS,
    ASPECT_TYPES,
    BIO_LABEL_O,
    NEG_HINTS,
    POS_HINTS,
    polarity_text,
)

ROOT = Path(__file__).resolve().parents[2]


def load_annotation_pool(
    path: str | Path = "data/samples/aspect_annotate_pool.csv",
) -> list[dict]:
    """上游: 杨国东 sample_for_annotation()；列 id, sentence。"""
    csv_path = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"标注池不存在: {csv_path}，请先运行 scripts/build_aspect_annotate_pool.py")

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            if not sentence:
                continue
            rows.append({"id": str(row.get("id", len(rows))), "sentence": sentence})
    return rows


def find_aspect_spans(sentence: str) -> list[tuple[int, int, str]]:
    """在句中找属性触发词 span。返回 [(start, end, aspect_type), ...]。"""
    spans: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for aspect in ASPECT_TYPES:
        for kw in ASPECT_KEYWORDS.get(aspect, (aspect,)):
            start = 0
            while True:
                idx = sentence.find(kw, start)
                if idx < 0:
                    break
                span = (idx, idx + len(kw))
                if span not in seen:
                    seen.add(span)
                    spans.append((idx, idx + len(kw), aspect))
                start = idx + 1
    return sorted(spans, key=lambda x: x[0])


def infer_polarity(sentence: str, aspect: str, sentence_label: int | None = None) -> int:
    """方面极性：局部上下文 + 句级 label 兜底。"""
    window = sentence
    for kw in ASPECT_KEYWORDS.get(aspect, (aspect,)):
        pos = sentence.find(kw)
        if pos >= 0:
            window = sentence[max(0, pos - 8) : pos + len(kw) + 8]
            break
    if any(h in window for h in NEG_HINTS):
        return 0
    if any(h in window for h in POS_HINTS):
        return 1
    if sentence_label is not None:
        return int(sentence_label)
    return 1


def tokenize_sentence(sentence: str) -> list[str]:
    return [t for t in jieba.lcut(sentence.strip()) if t.strip()]


def spans_to_token_labels(tokens: list[str], spans: list[tuple[int, int, str]], sentence: str) -> list[str]:
    """字符 span 对齐到 jieba 词级 BIO。"""
    labels = [BIO_LABEL_O] * len(tokens)
    if not tokens:
        return labels

    cursor = 0
    token_spans: list[tuple[int, int]] = []
    for tok in tokens:
        idx = sentence.find(tok, cursor)
        if idx < 0:
            idx = cursor
        token_spans.append((idx, idx + len(tok)))
        cursor = idx + len(tok)

    for start, end, aspect in spans:
        first = True
        for i, (ts, te) in enumerate(token_spans):
            if te <= start or ts >= end:
                continue
            labels[i] = f"B-{aspect}" if first else f"I-{aspect}"
            first = False
    return labels


def annotate_bio(
    record_id: str,
    sentence: str,
    aspect_spans: list[tuple[int, int, str]] | None = None,
    sentence_label: int | None = None,
) -> dict:
    """单句 BIO 标注 + 方面情感行。"""
    spans = aspect_spans if aspect_spans is not None else find_aspect_spans(sentence)
    tokens = tokenize_sentence(sentence)
    labels = spans_to_token_labels(tokens, spans, sentence)

    sentiment_rows: list[dict] = []
    aspects_in_span: set[str] = set()
    for _s, _e, aspect in spans:
        if aspect in aspects_in_span:
            continue
        aspects_in_span.add(aspect)
        pol = infer_polarity(sentence, aspect, sentence_label)
        sentiment_rows.append(
            {
                "id": record_id,
                "sentence": sentence,
                "aspect": aspect,
                "polarity": pol,
                "polarity_text": polarity_text(pol),
                "aspect_span": f"{_s}:{_e}",
            }
        )

    return {
        "id": record_id,
        "sentence": sentence,
        "tokens": tokens,
        "labels": labels,
        "sentiment_rows": sentiment_rows,
    }


def build_annotations_from_pool(
    pool: Iterable[dict],
    labels_by_id: dict[str, int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """批量标注。返回 (ner_records, sentiment_records)。"""
    ner_records: list[dict] = []
    sentiment_records: list[dict] = []
    labels_by_id = labels_by_id or {}

    for item in pool:
        rid = str(item["id"])
        sentence = item["sentence"]
        rec = annotate_bio(rid, sentence, sentence_label=labels_by_id.get(rid))
        ner_records.append(
            {"id": rec["id"], "tokens": rec["tokens"], "labels": rec["labels"], "sentence": rec["sentence"]}
        )
        sentiment_records.extend(rec["sentiment_rows"])

    return ner_records, sentiment_records

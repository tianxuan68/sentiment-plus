"""推理：predict_entities，供郑平高 AnalyzeResult.entities 对齐。"""

from __future__ import annotations

import json
from pathlib import Path

from pipelines.aspect_ner.annotate import annotate_bio, find_aspect_spans, infer_polarity
from pipelines.aspect_ner.constants import ASPECT_TYPES, polarity_text

ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = ROOT / "artifacts" / "aspect_ner" / "best"


def _rule_predict(text: str) -> list[dict]:
    """规则基线：关键词 span + 极性（Mock/降级与无 GPU 时使用）。"""
    sentence = text.strip()
    if not sentence:
        return []
    spans = find_aspect_spans(sentence)
    entities: list[dict] = []
    seen: set[str] = set()
    for start, end, aspect in spans:
        if aspect in seen:
            continue
        seen.add(aspect)
        pol = infer_polarity(sentence, aspect)
        entities.append(
            {
                "aspect": aspect,
                "text": sentence[start:end],
                "start": start,
                "end": end,
                "polarity": pol,
                "polarity_text": polarity_text(pol),
            }
        )
    if not entities:
        for aspect in ASPECT_TYPES:
            if aspect in sentence:
                idx = sentence.find(aspect)
                pol = infer_polarity(sentence, aspect)
                entities.append(
                    {
                        "aspect": aspect,
                        "text": aspect,
                        "start": idx,
                        "end": idx + len(aspect),
                        "polarity": pol,
                        "polarity_text": polarity_text(pol),
                    }
                )
                break
    return sorted(entities, key=lambda e: e["start"])


def _model_predict(text: str) -> list[dict] | None:
    """若已训练 BERT NER 权重则加载推理；否则返回 None。"""
    if not (_MODEL_DIR / "config.json").exists():
        return None
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError:
        return None

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(_MODEL_DIR)
    model.eval()

    tokens = list(text.strip())
    if not tokens:
        return []

    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    with torch.no_grad():
        logits = model(**enc).logits[0]
    pred_ids = logits.argmax(-1).tolist()
    id2label = model.config.id2label

    word_ids = enc.word_ids()
    labels: list[str] = []
    last_wid = None
    for i, wid in enumerate(word_ids):
        if wid is None or wid == last_wid:
            continue
        last_wid = wid
        if wid < len(tokens):
            labels.append(id2label.get(str(pred_ids[i]), "O"))

    # 词级转实体 span（字符级 tokens 即单字）
    entities: list[dict] = []
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab.startswith("B-"):
            aspect = lab[2:]
            start = i
            i += 1
            while i < len(labels) and labels[i] == f"I-{aspect}":
                i += 1
            end = i
            pol = infer_polarity(text, aspect)
            entities.append(
                {
                    "aspect": aspect,
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "polarity": pol,
                    "polarity_text": polarity_text(pol),
                }
            )
        else:
            i += 1
    return entities


def predict_entities(text: str, prefer_model: bool = True) -> list[dict]:
    """属性 NER 推理入口。
    入参: text — 单条评论
    返回: [{ aspect, text, start, end, polarity, polarity_text }, ...]
    """
    if prefer_model:
        model_out = _model_predict(text)
        if model_out is not None:
            return model_out
    return _rule_predict(text)


def load_metrics() -> dict:
    path = ROOT / "artifacts" / "metrics" / "ner_f1.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

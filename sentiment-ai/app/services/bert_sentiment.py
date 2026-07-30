"""加载 artifacts/bert/best 做情感推理（训练完成后可用）。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizer

AI_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = AI_ROOT / "artifacts" / "bert" / "best"
LABEL_MAP = {0: "负向", 1: "正向"}


@lru_cache(maxsize=1)
def _load(model_dir: str):
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 BERT 权重: {path}。请先完成微调并导出到 artifacts/bert/best"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path).to(device)
    model.eval()
    return tokenizer, model, device


def predict_sentiment(texts: list[str], model_dir: Path | None = None, max_length: int = 128) -> list[dict]:
    tokenizer, model, device = _load(str(model_dir or DEFAULT_MODEL_DIR))
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1).cpu().tolist()
        confs = probs.max(dim=-1).values.cpu().tolist()

    return [
        {
            "text": text,
            "label": pred,
            "label_name": LABEL_MAP.get(pred, str(pred)),
            "confidence": float(conf),
        }
        for text, pred, conf in zip(texts, preds, confs)
    ]

"""
英文情感本地预测（对齐 04_Bert / 中文 infer 思路）

用法:
    python -m pipelines.sentiment.bert.en.infer --text "Great product" "Worst purchase ever"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizer

AI_ROOT = Path(__file__).resolve().parents[4]
LABEL_MAP = {0: "Negative", 1: "Positive"}


def predict(
    texts: list[str],
    model_dir: Path,
    max_length: int = 128,
) -> list[dict]:
    """对英文句子列表做情感预测。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))
    model = BertForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

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

    results = []
    for text, pred, conf in zip(texts, preds, confs):
        results.append(
            {
                "text": text,
                "label": int(pred),
                "label_name": LABEL_MAP.get(pred, str(pred)),
                "confidence": float(conf),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="英文 BERT 本地推理")
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=AI_ROOT / "artifacts" / "bert_en" / "best",
    )
    parser.add_argument(
        "--text",
        type=str,
        nargs="+",
        default=["Works perfectly, highly recommend!", "Stopped working after one week."],
    )
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(
            f"未找到英文权重: {args.model_dir}\n请先运行: python -m pipelines.sentiment.bert.en.train"
        )

    for item in predict(args.text, args.model_dir, args.max_length):
        print(item)


if __name__ == "__main__":
    main()

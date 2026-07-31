"""本地推理自测（加载 artifacts/bert/best）。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizer

AI_ROOT = Path(__file__).resolve().parents[3]
LABEL_MAP = {0: "负向", 1: "正向"}


def predict(texts: list[str], model_dir: Path, max_length: int = 128) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
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

    return [
        {
            "text": text,
            "label": pred,
            "label_name": LABEL_MAP.get(pred, str(pred)),
            "confidence": float(conf),
        }
        for text, pred, conf in zip(texts, preds, confs)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT 本地推理")
    parser.add_argument("--model_dir", type=Path, default=AI_ROOT / "artifacts" / "bert" / "best")
    parser.add_argument("--text", type=str, nargs="+", default=["味道很好，会回购", "物流太慢，很失望"])
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"未找到权重: {args.model_dir}")

    for item in predict(args.text, args.model_dir, args.max_length):
        print(item)


if __name__ == "__main__":
    main()

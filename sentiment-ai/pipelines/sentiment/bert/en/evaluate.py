"""英文测试集评估：加载 artifacts/bert_en/best。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import BertForSequenceClassification, BertTokenizer

from .data_utils import build_dataloader
from .train import evaluate

AI_ROOT = Path(__file__).resolve().parents[4]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="英文 BERT 测试集评估")
    parser.add_argument(
        "--config",
        type=Path,
        default=AI_ROOT / "configs" / "bert_sentiment_en.yaml",
    )
    parser.add_argument("--model_dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    model_dir = args.model_dir or (AI_ROOT / cfg["output_dir"] / "best")
    test_csv = AI_ROOT / cfg["processed_dir"] / cfg["test_file"]
    if not model_dir.exists():
        raise FileNotFoundError(f"未找到权重: {model_dir}，请先 train")
    if not test_csv.exists():
        raise FileNotFoundError(f"未找到测试集: {test_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)

    loader = build_dataloader(
        test_csv,
        tokenizer,
        batch_size=int(cfg["batch_size"]),
        max_length=int(cfg["max_length"]),
        shuffle=False,
        max_samples=None,
        device=device,
    )
    metrics = evaluate(model, loader, device)
    print(f"[evaluate_en] Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f}")
    print(metrics["report"])

    out = AI_ROOT / cfg["output_dir"] / "test_metrics.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(
            {"accuracy": metrics["accuracy"], "f1": metrics["f1"]},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

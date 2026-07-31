"""
英文 BERT 情感微调（风格对齐 04_Bert：步骤清晰）

步骤:
    1. 读配置 bert_sentiment_en.yaml
    2. 用 data_utils 构建 train/val DataLoader
    3. 加载本地 bert-base-uncased + 分类头
    4. AdamW + warmup 训练
    5. 按验证集 F1 保存 artifacts/bert_en/best
    6. 写出 metrics.json

用法（在 sentiment-ai 根目录）:
    python -m pipelines.sentiment.bert.en.prepare_data
    python -m pipelines.sentiment.bert.en.split_data
    python -m pipelines.sentiment.bert.en.train
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from .data_utils import build_dataloader

AI_ROOT = Path(__file__).resolve().parents[4]
LABEL_NAMES = {0: "Negative", 1: "Positive"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model_path(model_name: str) -> str:
    """本地相对路径优先；保证读到 artifacts/pretrained/..."""
    p = Path(model_name)
    if not p.is_absolute():
        local = AI_ROOT / p
        if local.exists():
            return str(local)
    if not Path(model_name).exists() and "/" in model_name.replace("\\", "/"):
        raise FileNotFoundError(f"本地英文预训练模型不存在: {model_name}")
    return model_name


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """在 val/test loader 上算 Acc / F1。"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n = 0

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        bs = labels.size(0)
        total_loss += float(outputs.loss.item()) * bs
        n += bs

        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    n = max(n, 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return {
        "loss": total_loss / n,
        "accuracy": float(acc),
        "f1": float(f1),
        "report": classification_report(
            all_labels,
            all_preds,
            target_names=["Negative", "Positive"],
            digits=4,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="英文 BERT 情感微调")
    parser.add_argument(
        "--config",
        type=Path,
        default=AI_ROOT / "configs" / "bert_sentiment_en.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["random_seed"]))

    processed = AI_ROOT / cfg["processed_dir"]
    train_csv = processed / cfg["train_file"]
    val_csv = processed / cfg["val_file"]
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"未找到英文划分文件。请先运行:\n"
            f"  python -m pipelines.sentiment.bert.en.prepare_data\n"
            f"  python -m pipelines.sentiment.bert.en.split_data\n"
            f"期望: {train_csv}, {val_csv}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = resolve_model_path(cfg["model_name"])
    print(f"[train_en] device={device}")
    print(f"[train_en] model={model_name}")

    # tokenizer + 分类模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=int(cfg["num_labels"]),
    ).to(device)
    ln = model.bert.embeddings.LayerNorm.weight.detach().abs().mean().item()
    print(f"[train_en] LayerNorm.weight mean(abs)={ln:.4f} (正常约 0.5~1.0)")

    # DataLoader（04_Bert 风格 collate）
    train_loader = build_dataloader(
        train_csv,
        tokenizer,
        batch_size=int(cfg["batch_size"]),
        max_length=int(cfg["max_length"]),
        shuffle=True,
        max_samples=cfg.get("max_train_samples"),
        device=device,
    )
    val_loader = build_dataloader(
        val_csv,
        tokenizer,
        batch_size=int(cfg["batch_size"]),
        max_length=int(cfg["max_length"]),
        shuffle=False,
        max_samples=cfg.get("max_val_samples"),
        device=device,
    )
    print(
        f"[train_en] batches train={len(train_loader)} val={len(val_loader)} "
        f"(samples 受 max_*_samples 控制)"
    )

    epochs = int(cfg["epochs"])
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    total_steps = max(len(train_loader) * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(cfg.get("warmup_ratio", 0.1))),
        num_training_steps=total_steps,
    )

    out_dir = AI_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for input_ids, attention_mask, labels in pbar:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
        }
        history.append(record)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_dir = out_dir / "best"
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  -> saved best to {save_dir}")

    metrics_path = AI_ROOT / cfg["metrics_file"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"history": history, "best_val_f1": best_f1, "label_names": LABEL_NAMES},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[train_en] done. metrics -> {metrics_path}")


if __name__ == "__main__":
    main()

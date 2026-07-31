"""BERT 情感分类微调骨架（BertForSequenceClassification + AdamW）。

用法（在 sentiment-ai 根目录）:
  python -m pipelines.sentiment.bert.split_data
  python -m pipelines.sentiment.bert.train
"""
# 该模块实现了基于BERT的情感分类模型微调训练流程
# 使用BertForSequenceClassification作为基础模型，配合AdamW优化器进行训练
# 支持学习率预热、模型评估、早停保存最佳模型等功能

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from dataset import SentimentDataset

# sentiment-ai 根目录
AI_ROOT = Path(__file__).resolve().parents[3]


def set_seed(seed: int) -> None:
    """设置全局随机种子，确保实验可复现

    设置Python、NumPy和PyTorch的随机种子，使得每次运行代码时
    产生的随机数序列相同，从而保证实验结果的可复现性。

    Args:
        seed: 随机种子整数值
    """
    random.seed(seed)  # Python内置随机模块
    np.random.seed(seed)  # NumPy随机模块
    torch.manual_seed(seed)  # PyTorch CPU随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU随机种子（所有GPU）


def load_config(path: Path) -> dict:
    """加载YAML配置文件

    Args:
        path: 配置文件路径

    Returns:
        dict: 包含训练配置的字典
    """
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """在验证集上评估模型性能

    使用torch.no_grad()装饰器禁用梯度计算，减少内存消耗并加速推理。
    计算模型在验证集上的损失、准确率、F1分数和详细分类报告。

    Args:
        model: BERT分类模型
        loader: 验证集DataLoader
        device: 计算设备（CPU或GPU）

    Returns:
        dict: 包含以下评估指标的字典：
            - loss: 平均损失值
            - accuracy: 准确率
            - f1: 加权F1分数
            - report: sklearn分类报告（包含精确率、召回率等）
    """
    model.eval()  # 设置模型为评估模式
    all_preds, all_labels = [], []
    total_loss = 0.0

    for batch in loader:
        # 将batch数据移动到指定设备
        batch = {k: v.to(device) for k, v in batch.items()}

        # 前向传播，获取模型输出
        outputs = model(**batch)

        # 累加损失（乘以batch_size用于后续计算平均损失）
        total_loss += float(outputs.loss.item()) * batch["labels"].size(0)

        # 获取预测结果（取logits中概率最大的类别）
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()

        # 收集所有预测和标签
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    # 计算评估指标
    n = max(len(all_labels), 1)  # 避免除以零
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return {
        "loss": total_loss / n,
        "accuracy": float(acc),
        "f1": float(f1),
        "report": classification_report(all_labels, all_preds, digits=4),
    }


def main() -> None:
    """主函数：执行BERT情感分类模型的完整训练流程

    训练流程包括：
    1. 解析命令行参数并加载配置文件
    2. 设置随机种子确保可复现性
    3. 检查训练/验证数据文件是否存在
    4. 加载预训练BERT模型和分词器
    5. 创建数据集和数据加载器
    6. 配置优化器和学习率调度器
    7. 执行训练循环，每个epoch后进行验证
    8. 保存最佳模型（基于验证集F1分数）
    9. 保存训练历史和指标到JSON文件
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="BERT 情感分类微调")
    parser.add_argument(
        "--config",
        type=Path,
        default=AI_ROOT / "configs" / "bert_sentiment.yaml",
    )
    args = parser.parse_args()

    # 加载配置文件并设置随机种子
    cfg = load_config(args.config)
    set_seed(int(cfg["random_seed"]))

    # 检查训练和验证数据文件是否存在
    processed = AI_ROOT / cfg["processed_dir"]
    train_csv = processed / "train_split.csv"
    val_csv = processed / "val_split.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"未找到划分文件，请先运行: python -m pipelines.sentiment.bert.split_data\n"
            f"期望: {train_csv}, {val_csv}"
        )

    # 设置计算设备（优先使用GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 处理模型路径：支持本地路径和Hugging Face模型名称
    model_name = cfg["model_name"]
    model_path = Path(model_name)
    if not model_path.is_absolute():
        local = AI_ROOT / model_path
        if local.exists():
            model_name = str(local)
    if not Path(model_name).exists() and "/" in model_name.replace("\\", "/"):
        raise FileNotFoundError(f"本地预训练模型不存在: {model_name}")

    print(f"[train] device={device} model={model_name}")

    # 加载BERT分词器和分类模型
    # 优先加载目录内 model.safetensors；若仅有旧版 pytorch_model.bin
    # （LayerNorm.gamma/beta 键名），需先转换成新键名，否则 LayerNorm 等于随机初始化，Acc 会卡在 ~75%。
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=int(cfg["num_labels"]),  # 分类类别数
    ).to(device)
    ln = model.bert.embeddings.LayerNorm.weight.detach().abs().mean().item()
    print(f"[train] LayerNorm.weight mean(abs)={ln:.4f} (正常约 0.5~1.0；若接近 0.02 说明预训练未加载好)")

    # 创建训练和验证数据集
    train_ds = SentimentDataset(
        train_csv,
        tokenizer,
        max_length=int(cfg["max_length"]),
        max_samples=cfg.get("max_train_samples"),  # 可选：限制训练样本数
    )
    val_ds = SentimentDataset(
        val_csv,
        tokenizer,
        max_length=int(cfg["max_length"]),
        max_samples=cfg.get("max_val_samples"),  # 可选：限制验证样本数
    )
    print(f"[train] samples train={len(train_ds)} val={len(val_ds)}")

    # 创建数据加载器
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False)

    # 配置优化器和学习率调度器
    epochs = int(cfg["epochs"])
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.01)),  # L2正则化
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(cfg.get("warmup_ratio", 0.1))),  # 预热步数
        num_training_steps=total_steps,
    )

    # 初始化输出目录和训练状态
    out_dir = AI_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0  # 记录最佳F1分数
    history = []  # 训练历史记录

    # 训练循环
    for epoch in range(1, epochs + 1):
        model.train()  # 设置模型为训练模式
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")

        for batch in pbar:
            # 将batch数据移动到指定设备
            batch = {k: v.to(device) for k, v in batch.items()}
            print(batch)
            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss

            # 反向传播和参数更新
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            scheduler.step()  # 更新学习率

            # 记录损失
            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 计算平均训练损失
        train_loss = running / max(len(train_loader), 1)

        # 在验证集上评估模型
        val_metrics = evaluate(model, val_loader, device)

        # 记录本epoch的指标
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

        # 保存最佳模型（基于验证集F1分数）
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            model.save_pretrained(out_dir / "best")
            tokenizer.save_pretrained(out_dir / "best")
            print(f"  -> saved best to {out_dir / 'best'}")

    # 保存训练历史和最终指标到JSON文件
    metrics_path = AI_ROOT / cfg["metrics_file"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"history": history, "best_val_f1": best_f1}, f, ensure_ascii=False, indent=2)

    print(f"[train] done. metrics -> {metrics_path}")
    print("验收目标: test Acc>=0.92, F1>=0.90（请再跑 evaluate.py）")


if __name__ == "__main__":
    # 脚本入口点
    main()

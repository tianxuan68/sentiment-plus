"""
英文情感 DataLoader 工具（风格对齐 04_Bert/src/utils.py）

DataLoader 工作机制（复习）:
    1. Dataset 负责按索引返回「一条原始样本」(text, label)
    2. DataLoader 按 batch_size 取出若干条，交给 collate_fn
    3. collate_fn 用 BertTokenizer 批量编码，得到:
       input_ids, attention_mask, labels

步骤概览:
    1. load_samples: 读划分后的 CSV -> [(sentence, label), ...]
    2. TextDataset: 包装成 PyTorch Dataset
    3. collate_fn: tokenizer 编码一个 batch
    4. build_dataloader: 组装 DataLoader
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, PreTrainedTokenizerBase


class TextDataset(Dataset):
    """存放 (text, label) 列表，和投满分项目里的 TextDataset 一样。"""

    def __init__(self, data: list[tuple[str, int]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        # 返回评论文本 和 标签
        return self.data[idx][0], self.data[idx][1]


def load_samples(
    input_csv_path: str | Path,
    text_col: str = "sentence",
    label_col: str = "label",
    max_samples: int | None = None,
) -> list[tuple[str, int]]:
    """步骤1: 从 CSV 加载样本列表。

    参数:
        input_csv_path: 划分后的 CSV，例如 train_split_en.csv
        max_samples: 试跑时只取前 N 条；None 表示全量

    返回:
        [(text, label), ...]  其中 label 为 0/1
    """
    df = pd.read_csv(input_csv_path, encoding="utf-8-sig")
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"缺少列 {text_col}/{label_col}，当前={list(df.columns)}")

    df = df[[text_col, label_col]].dropna().copy()
    df[label_col] = df[label_col].astype(int)

    if max_samples is not None:
        df = df.head(int(max_samples)).reset_index(drop=True)

    data_list: list[tuple[str, int]] = []
    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        if not text:
            continue
        data_list.append((text, int(row[label_col])))
    return data_list


def encode_batch(
    data_batch: list[tuple[str, int]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """步骤3辅助: 把一个 batch 编成 BERT 输入（对齐 04_Bert encode_batch）。

    返回:
        input_ids, attention_mask, labels
    """
    texts = [item[0] for item in data_batch]
    labels = torch.tensor([item[1] for item in data_batch], dtype=torch.long, device=device)

    result = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = result["input_ids"].to(device)
    attention_mask = result["attention_mask"].to(device)
    return input_ids, attention_mask, labels


def build_dataloader(
    csv_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 16,
    max_length: int = 128,
    shuffle: bool = True,
    max_samples: int | None = None,
    device: str | torch.device | None = None,
) -> DataLoader:
    """步骤4: 组装英文情感 DataLoader。

    用法示例:
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        loader = build_dataloader('data/processed/train_split_en.csv', tokenizer)
        for input_ids, mask, labels in loader:
            ...
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # 1) 拿到数据列表
    data = load_samples(csv_path, max_samples=max_samples)
    # 2) Dataset
    dataset = TextDataset(data)

    # 3) collate_fn：和 04_Bert 一样，在这里做 tokenizer
    def collate_fn(batch: list[tuple[str, int]]):
        return encode_batch(batch, tokenizer, max_length=max_length, device=device)

    # 4) DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def demo_one_batch(csv_path: str | Path, bert_path: str | Path, batch_size: int = 2) -> None:
    """本地快速检查数据处理是否通。"""
    tokenizer = BertTokenizer.from_pretrained(str(bert_path))
    loader = build_dataloader(
        csv_path,
        tokenizer,
        batch_size=batch_size,
        max_length=128,
        shuffle=False,
        max_samples=8,
    )
    input_ids, mask, labels = next(iter(loader))
    print("[demo] input_ids:", tuple(input_ids.shape))
    print("[demo] mask     :", tuple(mask.shape))
    print("[demo] labels   :", labels.tolist())


if __name__ == "__main__":
    # 直接运行本文件时，做一次冒烟测试（需先跑 prepare + split）
    ai_root = Path(__file__).resolve().parents[4]
    csv_path = ai_root / "data" / "processed" / "train_split_en.csv"
    # 英文预训练路径；若还没有，请先准备 bert-base-uncased
    bert_path = ai_root / "artifacts" / "pretrained" / "bert-base-uncased"
    if not csv_path.exists():
        print(f"缺少 {csv_path}，请先运行 prepare_data.py 和 split_data.py")
    elif not bert_path.exists():
        print(f"缺少英文预训练目录: {bert_path}")
        print("可先完成划分；tokenizer 冒烟等模型就绪后再测。")
    else:
        demo_one_batch(csv_path, bert_path)

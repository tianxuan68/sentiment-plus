"""BERT 情感分类 Dataset 骨架。"""
# 该模块实现了PyTorch Dataset类，用于加载情感分析数据集
# 主要功能是读取CSV格式的数据，并使用BERT tokenizer进行文本编码
# 为BERT模型训练提供标准化的数据输入接口

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class SentimentDataset(Dataset):
    """读取划分后的 CSV（sentence, label），做 BertTokenizer 编码。

    该类继承自PyTorch的Dataset类，专门用于处理情感分析任务的数据集。
    支持从CSV文件加载数据，并通过BERT tokenizer将文本转换为模型可接受的输入格式。

    Attributes:
        df: pandas DataFrame，存储从CSV加载的原始数据
        tokenizer: BERT分词器，用于文本编码
        max_length: 文本序列的最大长度，超过此长度将被截断
        text_col: CSV中文本列的列名
        label_col: CSV中标签列的列名
    """

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        text_col: str = "sentence",
        label_col: str = "label",
        max_samples: int | None = None,
    ) -> None:
        """初始化SentimentDataset

        Args:
            csv_path: CSV数据文件的路径
            tokenizer: BERT预训练分词器实例
            max_length: 文本序列的最大长度，默认为128
            text_col: 文本列的列名，默认为"sentence"
            label_col: 标签列的列名，默认为"label"
            max_samples: 最大样本数量限制，用于调试或采样，默认为None表示使用全部数据

        Raises:
            ValueError: 如果CSV文件缺少必要的文本列或标签列
        """
        # 读取CSV数据文件
        self.df = pd.read_csv(csv_path)

        # 如果指定了最大样本数，只取前N条数据（常用于调试或快速验证）
        if max_samples is not None:
            self.df = self.df.head(int(max_samples)).reset_index(drop=True)

        # 存储tokenizer和配置参数
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col = text_col
        self.label_col = label_col

        # 验证CSV文件包含必要的列
        if text_col not in self.df.columns or label_col not in self.df.columns:
            raise ValueError(f"缺少列 {text_col}/{label_col}，当前={list(self.df.columns)}")

    def __len__(self) -> int:
        """返回数据集的样本总数

        Returns:
            int: 数据集中的样本数量
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """获取指定索引的样本数据

        从DataFrame中取出一行数据，使用BERT tokenizer对文本进行编码，
        并将标签转换为tensor格式。返回的字典可直接用于BERT模型训练。

        Args:
            idx: 样本索引

        Returns:
            dict: 包含以下键值对的字典：
                - input_ids: token化后的输入ID序列
                - attention_mask: 注意力掩码，标识哪些位置是真实token
                - token_type_ids: token类型ID（用于句子对任务）
                - labels: 情感标签tensor
        """
        # 获取指定索引的数据行
        row = self.df.iloc[idx]
        text = str(row[self.text_col])  # 获取文本内容
        label = int(row[self.label_col])  # 获取标签值

        # 使用BERT tokenizer对文本进行编码
        # truncation=True: 超过max_length时截断
        # padding="max_length": 不足max_length时填充到固定长度
        # return_tensors="pt": 返回PyTorch tensor格式
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # 移除batch维度，将[1, seq_len]的tensor转换为[seq_len]
        item = {k: v.squeeze(0) for k, v in encoded.items()}

        # 添加标签到返回字典中，使用long类型以适应CrossEntropyLoss
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

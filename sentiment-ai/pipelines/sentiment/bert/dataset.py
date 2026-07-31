"""
BERT 情感分类 Dataset
"""

# 1.导包
import pandas as pd
import torch
from torch.utils.data import Dataset


# 自定义dataset：1个继承3个重写
class SentimentDataset(Dataset):
    def __init__(
        self,
        csv_path,
        tokenizer,
        max_length=128,
        text_col='sentence',
        label_col='label',
        max_samples=None,
    ):
        # 重写1：初始化，读表
        self.df = pd.read_csv(csv_path)
        if max_samples is not None:
            self.df = self.df.head(int(max_samples)).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col = text_col
        self.label_col = label_col
        if text_col not in self.df.columns or label_col not in self.df.columns:
            raise ValueError(f'缺少列 {text_col}/{label_col}，当前={list(self.df.columns)}')

    def __len__(self):
        # 重写2：返回样本数
        return len(self.df)

    def __getitem__(self, idx):
        # 重写3：取一条并编码
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        label = int(row[self.label_col])
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

"""
定义BiLSTM + 多头attention 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from bilstm_multi_attention import MultiHeadAttention
from bilstm_config import Config


# ================================================================
# 一个继承：继承 nn.Module
# ================================================================
class BiLSTM_MultiHead(nn.Module):  # ← 继承 nn.Module

    # ================================================================
    # 重写1：__init__() → 定义网络层
    # ================================================================
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_heads=4, num_layers=2, dropout=0.5, pad_idx=0):
        super().__init__()  # ← 调用父类初始化

        # ---------- 第1层：词嵌入 ----------
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )

        # ---------- 第2层：BiLSTM ----------
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # ---------- 第3层：多头注意力 ----------
        self.multihead_attn = MultiHeadAttention(
            embed_dim=hidden_dim * 2,  # BiLSTM 双向拼接后维度
            num_heads=num_heads,
            dropout=0.1
        )

        # ---------- 第4层：分类层 ----------
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)


    # ================================================================
    # 重写2：forward() → 定义前向传播
    # ================================================================
    def forward(self, input_ids):
        """
        数据流动：
        (batch, seq) → (batch, seq, embed_dim) → (batch, seq, hidden*2)
        → (batch, seq, hidden*2) → (batch, hidden*2) → (batch, num_classes)
        """

        # ---------- 1. 词嵌入 ----------
        # (batch, seq) → (batch, seq, embed_dim)
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)

        # ---------- 2. BiLSTM ----------
        # (batch, seq, embed_dim) → (batch, seq, hidden*2)
        lstm_out, _ = self.lstm(embedded)

        # ---------- 3. 多头自注意力 ----------
        # (batch, seq, hidden*2) → (batch, seq, hidden*2)
        attn_out, attn_weights = self.multihead_attn(
            lstm_out,
            lstm_out,
            lstm_out
        )

        # ---------- 4. 取 [CLS] 位置 ----------
        # (batch, seq, hidden*2) → (batch, hidden*2)
        cls_vec = attn_out[:, 0, :]
        cls_vec = self.dropout(cls_vec)

        # ---------- 5. 分类 ----------
        # (batch, hidden*2) → (batch, num_classes)
        logits = self.fc(cls_vec)

        return logits, attn_weights

if __name__ == '__main__':
    # 实例化配置类
    config_obj = Config()
    model = BiLSTM_MultiHead(config_obj.vocab_size,
                             config_obj.embed_dim,
                             config_obj.hidden_dim,
                             config_obj.num_classes,
                             config_obj.num_heads,
                             config_obj.num_layers,
                             config_obj.dropout)
    result = model(torch.tensor([[2, 3, 4, 0, 0],
        [5, 6, 7, 8, 9]]))
    print(result)


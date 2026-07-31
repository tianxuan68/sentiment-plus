"""
定义多头自制力层和计算方法
"""

import torch
import torch.nn as nn
import math


def attention(q, k, v, mask=None, dropout=None):
    """
    自注意力核心计算函数

    Args:
        q: (batch, head, seq, d_k)
        k: (batch, head, seq, d_k)
        v: (batch, head, seq, d_k)
        mask: 掩码，可选
        dropout: dropout层，可选

    Returns:
        attn_value: 注意力结果 (batch, head, seq, d_k)
        attn_weight: 注意力权重 (batch, head, seq, seq)
    """
    # 1. 计算 Q × K^T / √d_k（亲密度矩阵）
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 掩码处理（可选）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. Softmax 变权重（每行和为1）
    attn_weight = torch.softmax(scores, dim=-1)

    # 4. Dropout（可选）
    if dropout is not None:
        attn_weight = dropout(attn_weight)

    # 5. 加权求和：权重 × V
    attn_value = torch.matmul(attn_weight, v)

    return attn_value, attn_weight


class MultiHeadAttention(nn.Module):
    """
    多头自注意力层
    多头 = 多个头并行计算注意力，然后拼接
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()

        # embed_dim 必须能被 num_heads 整除
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads  # 每个头的维度
        self.embed_dim = embed_dim

        # ===== 4 个线性层 =====
        # Q、K、V 投影 + 输出投影
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: (batch, seq, embed_dim)
            k: (batch, seq, embed_dim)
            v: (batch, seq, embed_dim)
            mask: 掩码 (batch, seq, seq) 或 None

        Returns:
            output: (batch, seq, embed_dim)
            attn_weight: (batch, num_heads, seq, seq)
        """
        batch_size = q.size(0)

        # ===== 1. 线性变换生成 Q、K、V =====
        Q = self.W_q(q)  # (batch, seq, embed_dim)
        K = self.W_k(k)  # (batch, seq, embed_dim)
        V = self.W_v(v)  # (batch, seq, embed_dim)

        # ===== 2. 分头 =====
        # (batch, seq, embed_dim) → (batch, num_heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # ===== 3. 计算注意力（每个头独立） =====
        attn_value, attn_weight = attention(Q, K, V, mask, self.dropout)
        # attn_value: (batch, num_heads, seq, d_k)
        # attn_weight: (batch, num_heads, seq, seq)

        # ===== 4. 合并头 =====
        # (batch, num_heads, seq, d_k) → (batch, seq, embed_dim)
        attn_value = attn_value.transpose(1, 2).contiguous()
        attn_value = attn_value.view(batch_size, -1, self.embed_dim)

        # ===== 5. 输出投影 =====
        output = self.W_o(attn_value)

        return output, attn_weight


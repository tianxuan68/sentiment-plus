# BiLSTM 情感分类 Pipeline 速查

## 模块拓扑

```
bilstm_config          ← 统一配置中心（路径、超参、设备）
bilstm_data_util       ← 分词工具（语言检测、中英分词、特殊标记）
bilstm_multi_attention ← 多头自注意力底层实现（attention函数 + MultiHeadAttention类）

bilstm_data_process    → 数据清洗、划分(8:1:1)、分词、保存到 data_process/
bilstm_build_vocab     → 从训练集构建词表，保存到 artifacts/sentiment/bilstm/vocab.json
bilstm_data_build_batch→ 文本→token ID序列、SentimentDataset、DataLoader
bilstm_model           → BiLSTM + MultiHeadAttention 模型定义（依赖 bilstm_multi_attention）
bilstm_model_train     → 训练主流程（含 Optuna 自动搜参、权重/指标/路演图保存）

bilstm_predict_fun     → 单条预测函数（加载权重→分词→推理）
bilstm_predict_api     → FastAPI 服务（/predict 端点）
bilstm_predict_api_test→ API 调用测试脚本
test                   → 数据质量检查（空行、误读float行）
```

## 执行顺序

```
1. bilstm_data_process   → 产出 data_process/{train,val,test}.csv + stats.json
2. bilstm_build_vocab    → 产出 artifacts/sentiment/bilstm/vocab.json
3. bilstm_model_train    → 训练（内部调用 bilstm_data_build_batch 加载数据和词表）
                          → 产出 权重 .pt + 指标 JSON + 路演图 .png
4. bilstm_predict_api    → 部署（内部调用 bilstm_predict_fun）
```

步骤 1 和 2 是训练的前置条件，步骤 4 需要步骤 3 产出的权重。

## 数据流向

```
原始数据 (data/train.csv + data/data.csv)
  │
  ▼ 清洗 → 划分(8:1:1) → 分词 → 保存
data_process/{train,val,test}.csv + stats.json
  │
  ▼ 训练集 → 统计词频 → 构建词表
artifacts/sentiment/bilstm/vocab.json
  │
  ▼ 文本→token ID→批次→训练
aspect_sentiment/bilstm/bilstm_sentiment_best.pt  (权重)
artifacts/metrics/bilstm/bilstm_sentiment_metrics.json  (指标)
pitch_assets/chen_jiangping/baseline_cnn_bilstm.png     (路演图)
```

## 关键路径一览

| 用途 | 路径（相对 sentiment_classification/） |
|------|--------------------------------------|
| 中文原始数据 | `data/train.csv`（sentence, label 列） |
| 英文原始数据 | `data/data.csv`（reviewText, overall 列） |
| 处理后数据 | `data_process/`（train/val/test.csv + stats.json） |
| 词表 | `../../../../artifacts/sentiment/bilstm/vocab.json` |
| 模型权重 | `../../../aspect_sentiment/bilstm/bilstm_sentiment_best.pt` |
| 指标 JSON | `../../../../artifacts/metrics/bilstm/bilstm_sentiment_metrics.json` |
| 路演图 | `../../../../pitch_assets/chen_jiangping/baseline_cnn_bilstm.png` |
| CNN 指标（对比用） | `../../../artifacts/metrics/cnn_metrics.json` |
| Baseline 指标（对比用） | `../../../artifacts/metrics/baseline_metrics.json` |

## 模型结构

```
输入 (batch, seq)
  → Embedding (vocab_size → embed_dim)
  → BiLSTM (embed_dim → hidden_dim×2, num_layers=2)
  → MultiHeadAttention (hidden_dim×2, num_heads=2)
  → 取 [CLS] 位置向量
  → Dropout → Linear(hidden_dim×2 → 2)
  → 二分类输出 (正向/负向)
```

特殊标记：`<PAD>=0, <UNK>=1, [CLS]=2, [SEP]=3`

## 训练关键参数（bilstm_config.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_len | 30 | 最大序列长度 |
| embed_dim | 32 | 词向量维度 |
| hidden_dim | 32 | LSTM 隐藏层维度 |
| num_layers | 2 | LSTM 层数 |
| num_heads | 2 | 多头注意力头数 |
| dropout | 0.5 | Dropout 比例 |
| learning_rate | 1e-3 | 学习率 |
| weight_decay | 0.1 | L2 正则化 |
| patience | 3 | 早停耐心值 |
| ecope | 10 | 每 trial 最大 epoch 数 |
| batch_size | 16 | 批次大小 |
| num_workers | 0 | DataLoader 工作进程（Windows 必须为 0） |
| zh_model | 'char' | 中文分词模式（char 逐字 / jieba 分词） |

## Optuna 搜索空间（bilstm_model_train.py → objective）

| 参数 | 搜索范围 | 类型 |
|------|---------|------|
| embed_dim | [128, 256, 512] | categorical |
| hidden_dim | [128, 256, 512] | categorical |
| num_layers | [1, 2, 3] | int |
| dropout | [0.3, 0.6] | float |
| learning_rate | [1e-4, 1e-2] | log-uniform |
| batch_size | [32, 64, 128] | categorical |

搜参结束后，最优参数自动写回 `bilstm_config.py` 并运行 final_run 做完整训练。

## 预测 API

- 端点：`POST /predict`
- 请求体：`{"text": "..."}`
- 返回：`{"text": "...", "pre_result": "好评/差评", "duration": ...}`
- 启动：直接运行 `bilstm_predict_api.py`，监听 `127.0.0.1:8000`

## 注意事项

1. **训练前置依赖**：必须先运行 `bilstm_data_process` 和 `bilstm_build_vocab`，再用 `bilstm_model_train`。
2. **Windows 兼容**：`num_workers=0` 是 Windows 下的必需设置。
3. **中文分词**：默认逐字（`zh_model='char'`），可选 jieba。
4. **数据清洗**：中英文数据都经过 `clean_texts()` 过滤无效行（空文本、NaN、纯空白）。
5. **词表管理**：支持增量更新（`get_vocab` 的 `is_create=False` 模式会加载已有词表并追加新词）。
6. **路演图目标**：BiLSTM F1 需比 CNN 高出 ≥2% 才算达标，图中会用绿色/红色标注。
7. **checkpoint 恢复**：`bilstm_predict_fun` 只加载 `state_dict`，不依赖 checkpoint 中的 config（但会尝试恢复）。
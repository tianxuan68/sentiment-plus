# 胡潇潇 · BERT 情感微调与主推理入口 · 任务细化

> 对齐主设计：商品评论情感分类 · BERT Fine-tuning；向 1 号提供推理入口  
> 全局约定见：[00-总览与全局约定.md](./00-总览与全局约定.md)

------------------------------------------------------------------------

# 一、岗位速览

| 项 | 内容 |
| -- | ---- |
| **人员** | 胡潇潇 |
| **主责** | BERT 情感微调与主推理 |
| **目录** | `pipelines/sentiment/bert/` + `app/`（推理入口） |
| **使用技术** | PyTorch、Transformer、`bert-base-chinese` |
| **验收标准** | ①完成 Fine-tuning；②Acc≥92%；③F1≥0.90；④优于 CNN/BiLSTM；⑤向 1 号提供推理入口 |
| **路演目录** | `sentiment-ai/pitch_assets/hu_xiaoxiao/` |

------------------------------------------------------------------------

# 二、输入

- 统一划分 + `sentence`/`label`（可用 `text_clean` 或原文，**组内固定一种并写进 configs**）
- CNN/BiLSTM 指标文件（用于证明「优于」）

------------------------------------------------------------------------

# 三、实现步骤（训练）

1. `BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)`
2. 超参（对齐主设计）：
   - `max_length=128`
   - `lr=2e-5`
   - `batch_size=16`
   - `epochs=2~4`
   - 优化器 AdamW
3. 验证集选优；测试集 Acc≥92%、F1≥0.90，且优于 CNN/BiLSTM。
4. 导出：
   - `artifacts/sentiment/bert/best/`（`config.json` + 权重 + tokenizer）
   - `artifacts/metrics/bert_sentiment_metrics.json`
5. 【路演】SOTA 指标卡 + **四模型**对比图 + 2～3 条样例。

------------------------------------------------------------------------

# 四、实现步骤（推理入口 · 交 1 号）

在 `sentiment-ai/app/` 落地最小可调用服务：

| 方法 | 路径 | 请求体 | 响应 `result` |
| ---- | ---- | ------ | ------------- |
| POST | `/api/v1/sentiment/predict` | `{ "text": "..." }` | `{ "label": 0\|1, "prob": float, "model": "bert" }` |
| GET | `/health` | — | `{ "status": "ok" }` |

建议文件：

```
app/
├── main.py                     # FastAPI 实例
├── api/sentiment.py            # 路由
├── services/bert_sentiment.py  # 加载权重、推理
└── schemas/sentiment.py        # Pydantic
```

本地启动示例：`uvicorn app.main:app --host 0.0.0.0 --port 8100`

------------------------------------------------------------------------

# 五、产出文件

| 路径 | 说明 |
| ---- | ---- |
| `artifacts/sentiment/bert/best/` | 微调权重 |
| `artifacts/metrics/bert_sentiment_metrics.json` | 指标 |
| `artifacts/metrics/compare_four_models.json` | Baseline/CNN/BiLSTM/BERT |
| `pitch_assets/hu_xiaoxiao/sota_card.png` | SOTA 卡 |
| `pitch_assets/hu_xiaoxiao/four_models.png` | 四模型图 |

------------------------------------------------------------------------

# 六、对接

| 交谁 | 交什么 |
| ---- | ------ |
| **郑平高（1号）** | 推理 Base URL、OpenAPI/`/docs`、示例 curl、模型版本号 |
| 6号 | `pitch_assets/hu_xiaoxiao/` |

------------------------------------------------------------------------

# 七、交付自检

| 检查项 | 达标 |
| ------ | ---- |
| Fine-tuning 完成 | ☐ |
| Acc≥92%；F1≥0.90 | ☐ |
| 优于 CNN/BiLSTM | ☐ |
| `/api/v1/sentiment/predict` 可调 | ☐ |
| 路演：SOTA 卡 + 四模型图 + 样例 | ☐ |

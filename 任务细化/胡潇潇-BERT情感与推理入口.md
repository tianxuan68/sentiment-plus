# 胡潇潇 · BERT 情感微调与主推理入口 · 任务细化

> 对齐主设计：商品评论情感分类 · BERT Fine-tuning；向 1 号提供推理入口  
> 情感对比链：**Baseline → BiLSTM → BERT**（三模型，不含 CNN）  
> 全局约定见：[00-总览与全局约定.md](./00-总览与全局约定.md)

------------------------------------------------------------------------

# 一、岗位速览

| 项 | 内容 |
| -- | ---- |
| **人员** | 胡潇潇 |
| **主责** | BERT 情感微调与主推理 |
| **目录** | `pipelines/sentiment/bert/` + `app/`（推理入口） |
| **使用技术** | PyTorch、Transformer、`bert-base-chinese` |
| **验收标准** | ①完成 Fine-tuning；②Acc≥92%；③F1≥0.90；④优于 Baseline/BiLSTM；⑤向 1 号提供推理入口 |
| **路演目录** | `sentiment-ai/pitch_assets/hu_xiaoxiao/` |

------------------------------------------------------------------------

# 二、输入

- 统一划分 + `sentence`/`label`（组内固定 sentence 或 text_clean）
- Baseline / BiLSTM 指标文件（用于证明「优于」）

------------------------------------------------------------------------

# 三、实现步骤（训练）

1. `BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)`
2. 超参：`max_length=128`，`lr=2e-5`，`batch_size=16`，`epochs=2~4`，AdamW
3. 验证集选优；测试集 Acc≥92%、F1≥0.90，且优于 Baseline/BiLSTM。
4. 导出 `artifacts/sentiment/bert/best/` + `artifacts/metrics/bert_sentiment_metrics.json`
5. 【路演】SOTA 指标卡 + **三模型**对比图 + 2～3 条样例。

------------------------------------------------------------------------

# 四、实现步骤（推理入口 · 交 1 号）

| 方法 | 路径 | 请求体 | 响应 `result` |
| ---- | ---- | ------ | ------------- |
| POST | `/api/v1/sentiment/predict` | `{ "text": "..." }` | `{ "label": 0\|1, "prob": float, "model": "bert" }` |
| POST | `/api/v1/sentiment/models/compare` | `{}` | `ModelCompareResult`（三模型） |
| GET | `/health` | — | `{ "status": "ok" }` |

------------------------------------------------------------------------

# 五、函数级接口契约

```python
def train_bert(config: dict) -> dict:
    """返回: { acc, f1, model_dir, metrics_path }；acc≥0.92, f1≥0.90"""

def build_model_compare(
    baseline_path: str = "artifacts/metrics/baseline_metrics.json",
    bilstm_path: str = "artifacts/metrics/bilstm_sentiment_metrics.json",
    bert_path: str = "artifacts/metrics/bert_sentiment_metrics.json",
) -> dict:
    """三模型对比 JSON。
    返回: {
        "metrics": [{model, acc, f1, owner}, ...],  # 3 条
        "best_model": str,
        "bilstm_vs_baseline_f1_gain": float,
    }
    上游: 毛鑫泽 export_best()、陈江平 train_bilstm_sentiment()
    下游: 郑平高 get_model_compare()
    """

def predict_sentiment(text: str) -> dict:
    """返回: { label, prob, model: "bert" }"""

def create_app() -> FastAPI:
    """注册 /health、/sentiment/predict、/sentiment/models/compare 等"""
```

------------------------------------------------------------------------

# 六、产出文件

| 路径 | 说明 |
| ---- | ---- |
| `artifacts/sentiment/bert/best/` | 微调权重 |
| `artifacts/metrics/bert_sentiment_metrics.json` | 指标 |
| `artifacts/metrics/compare_three_models.json` | Baseline/BiLSTM/BERT |
| `pitch_assets/hu_xiaoxiao/sota_card.png` | SOTA 卡 |
| `pitch_assets/hu_xiaoxiao/three_models.png` | 三模型图 |

------------------------------------------------------------------------

# 七、对接

| 交谁 | 交什么 |
| ---- | ------ |
| **郑平高（1号）** | 推理 Base URL、OpenAPI、示例 curl |
| 6号 | `pitch_assets/hu_xiaoxiao/` |

------------------------------------------------------------------------

# 八、交付自检

| 检查项 | 达标 |
| ------ | ---- |
| Fine-tuning 完成 | ☐ |
| Acc≥92%；F1≥0.90 | ☐ |
| 优于 Baseline/BiLSTM | ☐ |
| `/api/v1/sentiment/predict` 可调 | ☐ |
| 路演：SOTA 卡 + 三模型图 + 样例 | ☐ |

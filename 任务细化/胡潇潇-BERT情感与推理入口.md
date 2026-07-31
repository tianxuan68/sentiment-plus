# 胡潇潇 · BERT 情感 + 推理入口（手把手）

> 你是对比链第 3 环（最强模型），并且要搭好 **AI 推理服务入口**，方便郑平高对接。  
> 详细字段：见 [01-函数接口对照表.md](./01-函数接口对照表.md)

---

## 1. 你在整条链上的位置

```
刘攀数据 + 共用 splits
毛鑫泽 / 陈江平 的指标 JSON
        ↓
【你】BERT 微调（Acc≥92%, F1≥0.90）
        ↓
【你】sentiment-ai/app 推理服务
   /health
   /api/v1/sentiment/predict
   /api/v1/sentiment/models/compare
        ↓
      郑平高转发到前端页面
```

**一句话**：训练出最好模型，并让 1 号「一个 Base URL」就能调通。

---

## 2. 你要完成的功能

| # | 功能 | 做到什么算完 |
| - | ---- | ------------ |
| 1 | BERT Fine-tuning | `bert-base-chinese`，2 分类 |
| 2 | Acc≥**92%**，F1≥**0.90** | 且优于 Baseline、BiLSTM |
| 3 | 三模型对比 JSON | 读毛鑫泽+陈江平指标拼出来 |
| 4 | 推理 HTTP | predict + compare + health |
| 5 | 路演图 | SOTA 卡 + 三模型图 |

---

## 3. 和谁对接

| 方向 | 找谁 | 干什么 |
| ---- | ---- | ------ |
| 上游数据 | **刘攀** | 语料 + `splits/` |
| 上游指标 | **毛鑫泽、陈江平** | 两份 metrics JSON |
| 下游主对接 | **郑平高** | 给 Base URL（如 `http://127.0.0.1:8100`）+ `/docs` + curl 示例 |
| 同 app 挂路由 | **杨国东、邓新晓、陈江平** | 关键词/优缺点/属性路由可挂你搭的 app |
| 交路演图 | **6号** | `pitch_assets/hu_xiaoxiao/` |

---

## 4. 传入 → 技术处理 → 返回

| 阶段 | 内容 |
| ---- | ---- |
| **传入（训练）** | `sentence`/`text_clean` + `label` + 统一 splits |
| **中间技术** | PyTorch + Transformers：`BertForSequenceClassification`；`max_length=128`，`lr=2e-5`，`batch=16`，`epochs=2~4`，AdamW |
| **返回（推理）** | 见下表 |

| 接口 | 传入 | 返回 |
| ---- | ---- | ---- |
| `GET /health` | 无 | `{status:"ok"}` |
| `POST /api/v1/sentiment/predict` | `{text}` | `{label:0\|1, prob, model:"bert"}` |
| `POST /api/v1/sentiment/models/compare` | `{}` | `{metrics:[{model,acc,f1,owner}×3], best_model, bilstm_vs_baseline_f1_gain}` |

统一外壳：`{code:0, msg:"ok", result:{...}}`

---

## 5. 怎么做（按顺序勾）

1. 训练代码：`pipelines/sentiment/bert/`（可参考已有 train/evaluate/infer）  
2. 权重导出：`artifacts/sentiment/bert/best/`  
3. 指标：`artifacts/metrics/bert_sentiment_metrics.json`  
4. 拼三模型：`build_model_compare()` → `compare_three_models.json`  
5. 启动 `app/`（FastAPI），至少暴露 health + predict + compare  
6. 给郑平高一份 curl，让他把 `.env` 里 `SENTIMENT_AI_MOCK=false` 后能通  
7. 路演图 ≥1920 宽

**给郑平高的 curl 示例（你交接口时附上）：**

```bash
curl -X POST http://127.0.0.1:8100/api/v1/sentiment/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"物流很快，包装完好\"}"
```

---

## 6. 验收打勾

| 检查 | ☐ |
| ---- | - |
| Fine-tuning 完成；Acc≥92%；F1≥0.90 | ☐ |
| 明确优于 Baseline / BiLSTM | ☐ |
| `/health` 与 `/sentiment/predict` 郑平高能调 | ☐ |
| 三模型对比可返回 | ☐ |
| 路演：SOTA 卡 + 三模型图已交 6 号 | ☐ |

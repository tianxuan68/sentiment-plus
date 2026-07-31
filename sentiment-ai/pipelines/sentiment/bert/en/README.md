# 英文 BERT 情感分类（胡潇潇）

## 流程

```powershell
cd sentiment-ai

# 数据（若已跑过可跳过）
python -m pipelines.sentiment.bert.en.prepare_data
python -m pipelines.sentiment.bert.en.split_data

# 训练（配置见 configs/bert_sentiment_en.yaml）
python -m pipelines.sentiment.bert.en.train

# 测试集评估
python -m pipelines.sentiment.bert.en.evaluate

# 命令行预测
python -m pipelines.sentiment.bert.en.infer --text "Great product" "Terrible quality"

# HTTP（交 1 号）
uvicorn app.main:app --reload --port 8100
# POST /api/sentiment/predict_en  {"texts":["Great product"]}
```

## 预训练

本地目录：`artifacts/pretrained/bert-base-uncased/`（已拉取，含 `model.safetensors`）

## 配置要点

- `max_train_samples` / `max_val_samples`：试跑用小数；全量改 `null`
- 输出权重：`artifacts/bert_en/best/`
- 与中文隔离：不要写入 `artifacts/bert/`

## 标签

| label | 含义 |
|-------|------|
| 0 | Negative |
| 1 | Positive |

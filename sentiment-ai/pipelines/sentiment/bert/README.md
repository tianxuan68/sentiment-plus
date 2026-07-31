# BERT 情感分类微调（胡潇潇）

## 目录职责

- 训练 / 划分 / 评估：本目录
- 推理 HTTP：`sentiment-ai/app/`
- 数据：`data/raw/train.csv`（中文主训），划分结果 → `data/processed/`
- 权重：`artifacts/bert/best/`（不入库）
- 预训练：`artifacts/pretrained/bert-base-chinese/`（本地拷贝，不联网下载）

## 重要：Git 不会带上的东西

以下文件被 `.gitignore` 忽略，**推送代码后队友拿不到**，需你单独发给他们（网盘 / U 盘 / 共享盘）：

| 内容 | 放到队友机器上的路径 |
|------|----------------------|
| 中文数据 `train.csv` | `sentiment-ai/data/raw/train.csv` |
| 预训练模型目录 `bert-base-chinese/`（含 `pytorch_model.bin` 等） | `sentiment-ai/artifacts/pretrained/bert-base-chinese/` |

英文 `data.csv` 本任务不混训，可不发。

## 环境（有 GPU 的队友）

```powershell
cd sentiment-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r pipelines\sentiment\bert\requirements.txt
# 若有 NVIDIA GPU，建议再装对应 CUDA 版 torch，例如：
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

确认 GPU：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

应打印 `True`。

## 配置检查（正式全量训练前）

打开 `configs/bert_sentiment.yaml`，确认：

- `max_train_samples: null`
- `max_val_samples: null`
- `epochs: 2`（或 3～4）
- `model_name: artifacts/pretrained/bert-base-chinese`

## 训练流程（在 sentiment-ai 根目录）

```powershell
python -m pipelines.sentiment.bert.split_data
python -m pipelines.sentiment.bert.train
python -m pipelines.sentiment.bert.evaluate
python -m pipelines.sentiment.bert.infer --text "味道很好" "物流太慢"
```

训练结束后把 `artifacts/bert/best/` 整包发回给你（用于推理 / 交 1 号）。

## 启动推理服务

```powershell
uvicorn app.main:app --reload --port 8100
```

## 超参

本地 `bert-base-chinese`，`max_length=128`，`lr=2e-5`，`batch_size=16`，`epochs=2~4`，AdamW

## 验收

测试集 Acc ≥ 92%，F1 ≥ 0.90，优于 CNN/BiLSTM 后导出权重。

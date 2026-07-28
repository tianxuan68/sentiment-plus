# 5号 · 属性实体识别（aspect_ner）

目录：`pipelines/aspect_ner/`（对齐 README 分工）

## 产出

| 文件 | 说明 |
| ---- | ---- |
| `data/processed/aspect_ner.jsonl` | BIO 序列标注 |
| `data/processed/aspect_sentiment.csv` | 方面 + polarity |
| `artifacts/metrics/ner_f1.json` | 实体 F1 |
| `pitch_assets/no05_ner/*.png` | 路演三图 |

## 快速运行

```powershell
cd sentiment-ai

# 1. 生成标注池（≥3000 句，seed=68；杨国东交付后可跳过）
python scripts/build_aspect_annotate_pool.py

# 2. 标注 + 导出 + 基线评估 + 路演图（默认跳过 BERT 训练）
python pipelines/aspect_ner/run_pipeline.py --skip-train

# 3. 可选：安装 ML 依赖后训练 BertForTokenClassification
pip install -r requirements-ml.txt
python pipelines/aspect_ner/run_pipeline.py

# 4. 单元测试
python tests/test_aspect_ner.py
```

## 核心函数

- `load_annotation_pool()` ← `data/samples/aspect_annotate_pool.csv`
- `annotate_bio()` / `build_annotations_from_pool()`
- `export_ner_jsonl()` / `export_aspect_sentiment()`
- `predict_entities(text)` → 郑平高 `AspectEntity` 对齐
- `train_ner()` → `BertForTokenClassification`

详见：[任务细化/5号-属性实体识别与标注.md](../任务细化/5号-属性实体识别与标注.md)

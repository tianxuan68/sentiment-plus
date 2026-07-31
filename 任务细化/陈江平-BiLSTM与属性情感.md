# 陈江平 · BiLSTM 情感 + 属性情感（手把手）

> 你有两件事：① 整句情感 BiLSTM（对比链第 2 环）；② 属性级情感。  
> 情感对比链：**Baseline → 你(BiLSTM) → BERT**  
> 详细字段：见 [01-函数接口对照表.md](./01-函数接口对照表.md)

---

## 1. 你在整条链上的位置

```
刘攀 clean + 共用 splits
毛鑫泽 baseline_metrics.json（你要超过他的 F1）
        ↓
【你·A】BiLSTM+Attention 整句情感
        ↓
   指标交给胡潇潇画三模型图

5号 aspect_sentiment.csv
        ↓
【你·B】BiLSTM 属性情感
        ↓
   权重/接口 → 郑平高（或与属性BERT共用路径）
```

---

## 2. 你要完成的功能

### 任务 A · 整句情感

| # | 功能 | 做到什么算完 |
| - | ---- | ------------ |
| 1 | BiLSTM + Attention | 双向语义建模 |
| 2 | F1 ≥ **0.88** | 测试集 |
| 3 | 比 Baseline F1 **+≥2%** | 写进对比 JSON |
| 4 | 对比图 | Baseline vs BiLSTM |

### 任务 B · 属性情感

| # | 功能 | 做到什么算完 |
| - | ---- | ------------ |
| 1 | 属性极性分类 | 用 5 号的表训练 |
| 2 | F1 ≥ **85%** | |
| 3 | 导出指定文件名 | `bilstm_aspect_best.pt` |

---

## 3. 和谁对接

| 方向 | 找谁 | 干什么 |
| ---- | ---- | ------ |
| 上游清洗 | **刘攀** | `clean_train.csv` + `splits/` |
| 上游尺子 | **毛鑫泽** | `baseline_metrics.json` 里的 best.f1 |
| 上游属性样本 | **5号** | `aspect_sentiment.csv` |
| 下游三模型 | **胡潇潇** | 你的情感指标 JSON |
| 下游展示 | **郑平高** | 属性预测接口（或权重说明） |
| 交路演图 | **6号** | 双模型对比图 + 属性 F1 卡 |

---

## 4. 传入 → 技术处理 → 返回

### 任务 A

| 阶段 | 内容 |
| ---- | ---- |
| **传入** | `text_clean`/`sentence` + `label`；同一 `splits/` |
| **中间技术** | PyTorch：Embedding → BiLSTM → Attention → Linear（2 类） |
| **返回** | `{label, prob, model:"bilstm"}`；权重 `.pt`；指标 JSON |

硬指标：`f1 ≥ 0.88` 且 `f1 - baseline_f1 ≥ 0.02`

### 任务 B

| 阶段 | 内容 |
| ---- | ---- |
| **传入** | `sentence` + `aspect` + `polarity(0/1)` |
| **中间技术** | 句+方面编码 → BiLSTM+Attention → 极性 |
| **返回** | `{aspect, polarity, polarity_text, prob}` |

**HTTP（与属性 BERT 共用路径）：**

| 接口 | 传入 | 返回 |
| ---- | ---- | ---- |
| `POST /api/v1/aspect/predict` | `{text, aspects:["质量","物流",...]}` | `{items:[{aspect,polarity,polarity_text,prob}]}` |

---

## 5. 怎么做（按顺序勾）

**任务 A**

1. 代码：`pipelines/sentiment/bilstm/`  
2. 只用训练集建词表，保存 `vocab.json`  
3. 训 BiLSTM+Attention，早停看 `val_f1`  
4. 导出 `artifacts/sentiment/bilstm/bilstm_sentiment_best.pt`  
5. 写 `artifacts/metrics/bilstm_sentiment_metrics.json` + 对比图  

**任务 B**

1. 代码：`pipelines/aspect_sentiment/bilstm/`  
2. 读 5 号 CSV，`seed=68` 划分（可单独 `splits_aspect/`）  
3. 训到 F1≥0.85  
4. 导出 **`artifacts/aspect/bilstm_aspect_best.pt`**（文件名固定）  
5. 路演 F1 卡 → `pitch_assets/chen_jiangping/`

---

## 6. 验收打勾

| 检查 | ☐ |
| ---- | - |
| BiLSTM+Attention 情感已实现 | ☐ |
| 情感 F1≥0.88 且比 Baseline +≥2% | ☐ |
| `bilstm_aspect_best.pt`；属性 F1≥85% | ☐ |
| 指标已给胡潇潇；路演图已交 6 号 | ☐ |

# 待定 · BERT 属性级情感（手把手）

> 岗位置空：人到位后改文件名即可。  
> 你的工作 = 用 BERT **句对**做「一句评论 × 多个属性」的情感，接口交 1 号。

---

## 1. 你在整条链上的位置

```
5号 aspect_sentiment.csv
        ↓
【你】BERT 句对微调：[CLS] 句子 [SEP] 属性 [SEP]
        ↓
  POST /api/v1/aspect/predict（可与陈江平 BiLSTM 共用路径）
        ↓
      郑平高单条分析页的 aspects[]
```

---

## 2. 你要完成的功能

| # | 功能 | 做到什么算完 |
| - | ---- | ------------ |
| 1 | 属性情感微调 | Acc ≥ **90%** |
| 2 | 一句多属性 | `predict_aspects(text, aspects[])` |
| 3 | 接口交 1 号 | 字段对齐 `AspectItem` |
| 4 | 路演 | 多属性面板截图 |

---

## 3. 和谁对接

| 方向 | 找谁 | 干什么 |
| ---- | ---- | ------ |
| 上游要样本 | **5号** | `aspect_sentiment.csv`（`sentence,aspect,polarity`） |
| 同数据对照 | **陈江平** | 同一份 CSV，方便比 BiLSTM vs BERT |
| 下游交接口 | **郑平高** | `/api/v1/aspect/predict` |
| 交路演图 | **6号** | 面板截图 |

---

## 4. 传入 → 技术处理 → 返回

| 阶段 | 内容 |
| ---- | ---- |
| **传入** | `text` + `aspects: ["质量","价格","物流",...]` |
| **中间技术** | `bert-base-chinese`；句对编码；二分类（正/负） |
| **返回** | `{items:[{aspect, polarity, polarity_text, prob}]}`，长度 = 属性个数 |

| 字段 | 含义 |
| ---- | ---- |
| `aspect` | 属性名 |
| `polarity` | 0/1 |
| `polarity_text` | 正向/负向 |
| `prob` | 概率 |

---

## 5. 怎么做（按顺序勾）

1. 代码放：`pipelines/aspect_sentiment/bert/`  
2. 编码：`[CLS] sentence [SEP] aspect [SEP]`  
3. 训练到 Acc≥90%，权重 `artifacts/aspect/bert_aspect_best/`  
4. 实现 `predict_aspects`，挂同一 HTTP 路径  
5. 给郑平高 curl；确认 `items` 字段名  
6. 路演截图

---

## 6. 验收打勾

| 检查 | ☐ |
| ---- | - |
| Acc≥90% | ☐ |
| 多属性返回条数 = 请求 aspects 长度 | ☐ |
| 郑平高能接入单条分析 | ☐ |
| 路演截图已交 | ☐ |

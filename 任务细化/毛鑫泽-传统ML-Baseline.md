# 毛鑫泽 · 传统 ML Baseline（手把手）

> 你的工作 = **对比链第 1 环**：用 TF-IDF + 传统模型证明「最基础能做到多少」。  
> 详细字段：见 [01-函数接口对照表.md](./01-函数接口对照表.md)

---

## 1. 你在整条链上的位置

```
刘攀 clean_train.csv
        ↓
   【你】TF-IDF + KNN/决策树/随机森林
        ↓
  交出指标 JSON + 最佳模型
        ↓
陈江平要比你的 F1 高 ≥2%；胡潇潇画三模型图也要你的数
```

**一句话**：你是 Baseline，后面所有「提升了多少」都以你为尺子。

---

## 2. 你要完成的功能

| # | 功能 | 做到什么算完 |
| - | ---- | ------------ |
| 1 | 至少 3 个传统模型 | KNN + 决策树 + 随机森林 |
| 2 | 同一测试集评估 | Acc / P / R / F1 |
| 3 | Acc ≥ **85%** | 选出最佳并导出 |
| 4 | 路演图 | 三模型柱状图 + 指标表 |

---

## 3. 和谁对接

| 方向 | 找谁 | 干什么 |
| ---- | ---- | ------ |
| 上游要表 | **刘攀** | 要 `clean_train.csv`（读 `text_clean`、`label`） |
| 上游要划分 | **全组约定** | 必须用 `data/processed/splits/`，`seed=68` |
| 下游交指标 | **陈江平、胡潇潇** | `baseline_metrics.json` |
| 交路演图 | **6号** | 柱状图、指标表 PNG |

---

## 4. 传入 → 技术处理 → 返回

| 阶段 | 内容 |
| ---- | ---- |
| **传入** | `text_clean` 列表 + `label`；按 `splits/` 分成 train/val/test |
| **中间技术** | `TfidfVectorizer(max_features=20000, ngram_range=(1,2))`（**只在 train 上 fit**）→ sklearn 三个分类器 |
| **返回/产出** | 最佳模型文件 + 指标 JSON |

**指标 JSON 长这样（字段名别改）：**

```json
{
  "models": [
    { "name": "knn", "acc": 0.86, "precision": 0.85, "recall": 0.84, "f1": 0.845 }
  ],
  "best_model": "rf",
  "best": { "acc": 0.872, "f1": 0.861 }
}
```

**单条预测（可选）返回：**

| 字段 | 含义 |
| ---- | ---- |
| `label` | 0 或 1 |
| `prob` | 概率 |
| `model` | 如 `"baseline_rf"` |

---

## 5. 怎么做（按顺序勾）

1. 代码放：`sentiment-ai/pipelines/sentiment/baseline/`
2. `load_corpus()`：读 clean 表 + 按 splits 过滤（禁止自己 `train_test_split`）
3. 只在训练集 `fit` TF-IDF
4. 训 KNN / DT / RF，在**同一 test** 上算指标
5. Acc≥85% 的最佳模型导出：
   - `artifacts/sentiment/baseline/best_model.joblib`
   - `artifacts/metrics/baseline_metrics.json`
6. 画图 → `pitch_assets/mao_xinze/`

| 函数 | 传入 | 返回 |
| ---- | ---- | ---- |
| `load_corpus(split)` | 路径 + train/val/test | `(texts, labels)` |
| `train_classifier(X,y,name)` | 特征、标签、模型名 | 已训练分类器 |
| `evaluate(y_true,y_pred)` | 真值、预测 | `{acc,precision,recall,f1}` |
| `export_best(...)` | 模型+指标 | 写出 joblib + json |

---

## 6. 验收打勾

| 检查 | ☐ |
| ---- | - |
| ≥3 种传统模型 | ☐ |
| 测试集 Acc≥85% | ☐ |
| `baseline_metrics.json` 已给陈江平/胡潇潇 | ☐ |
| 路演柱状图+表已交 6 号 | ☐ |

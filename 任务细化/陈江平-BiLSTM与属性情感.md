# 陈江平 · BiLSTM 情感 + 属性情感 · 任务细化

> 对齐主设计：商品评论情感分类（LSTM/BiLSTM）+ 商品属性情感分析（属性与情感关系建模）  
> **TextCNN**：主设计未单列责任人，由本岗在对比实验中自建轻量 TextCNN，保证「相对 CNN +2% F1」「四模型对比」可验收。  
> 全局约定见：[00-总览与全局约定.md](./00-总览与全局约定.md)

------------------------------------------------------------------------

# 一、岗位速览

| 项 | 内容 |
| -- | ---- |
| **人员** | 陈江平 |
| **主责** | BiLSTM 情感 + 属性情感；对照 CNN「+2% F1」 |
| **目录** | `pipelines/sentiment/bilstm/`；`pipelines/sentiment/cnn/`；`pipelines/aspect_sentiment/bilstm/` |
| **使用技术** | PyTorch、Embedding、BiLSTM、Attention |
| **路演目录** | `sentiment-ai/pitch_assets/chen_jiangping/` |

------------------------------------------------------------------------

# 二、任务 A · LSTM/BiLSTM 上下文语义建模（情感）

| 项 | 内容 |
| -- | ---- |
| **验收标准** | ①双向语义建模；②F1≥0.88；③相对 CNN 的 F1 提升≥2%；④完成 Baseline/CNN/BiLSTM 三方对比 |

## 2.1 输入

- 同一 `splits/` + `clean_train.csv` / `sentence`
- 毛鑫泽 `baseline_metrics.json`（写进对比报告）

## 2.2 实现步骤

1. **词表**：仅用训练集建 vocab；`<pad>`/`<unk>`；保存 `artifacts/sentiment/bilstm/vocab.json`。
2. **TextCNN（对照）** 于 `pipelines/sentiment/cnn/`：
   - Embedding + 多核 Conv1d + MaxPool + Linear
   - 同一划分、同一评估脚本；导出 `artifacts/metrics/cnn_metrics.json`
3. **BiLSTM+Attention**：
   - Embedding → BiLSTM → Attention 加权 → Linear → 2 类
   - 早停：monitor=`val_f1`，patience 组内自定（建议 3～5）
4. 测试集 F1≥0.88，且 `BiLSTM_F1 − CNN_F1 ≥ 0.02`。
5. 导出权重 `artifacts/sentiment/bilstm/bilstm_sentiment_best.pt` + metrics。
6. 写三方对比：`artifacts/metrics/compare_baseline_cnn_bilstm.json` + 路演图（标「+2% F1」）。

## 2.3 产出文件

| 路径 | 说明 |
| ---- | ---- |
| `artifacts/sentiment/cnn/cnn_best.pt` | TextCNN 权重 |
| `artifacts/metrics/cnn_metrics.json` | CNN 指标 |
| `artifacts/sentiment/bilstm/bilstm_sentiment_best.pt` | BiLSTM 情感权重 |
| `artifacts/metrics/bilstm_sentiment_metrics.json` | BiLSTM 指标 |
| `pitch_assets/chen_jiangping/baseline_cnn_bilstm.png` | 三方对比图 |

## 2.4 对接（情感）

| 交谁 | 交什么 |
| ---- | ------ |
| 胡潇潇 | CNN/BiLSTM 指标，便于四模型总图 |
| 6号 | 情感对比图 |

------------------------------------------------------------------------

# 三、任务 B · 属性与情感关系建模

| 项 | 内容 |
| -- | ---- |
| **验收标准** | ①统一划分训练测试；②F1≥85%；③导出 `bilstm_aspect_best.pt` |

## 3.1 输入

- 5 号交付的 `data/processed/aspect_sentiment.csv`  
  建议列：`sentence`、`aspect`、`polarity`（0/1 或 neg/pos，组内统一）

## 3.2 实现步骤

1. 按 seed=68 对属性样本划分 train/val/test（可单独 `splits_aspect/`，但种子必须 68）。
2. 模型：句 + 方面信息编码（拼接 aspect token 或独立 embedding）→ BiLSTM + Attention → 极性分类。
3. 评估 `f1_score`；目标 **≥0.85**。
4. 导出 **`artifacts/aspect/bilstm_aspect_best.pt`**（文件名与主设计一致）。
5. 【路演】F1 数字卡 + 2～3 条预测样例 PNG。

## 3.3 产出文件

| 路径 | 说明 |
| ---- | ---- |
| `artifacts/aspect/bilstm_aspect_best.pt` | 主设计指定文件名 |
| `artifacts/metrics/bilstm_aspect_metrics.json` | F1 等 |
| `pitch_assets/chen_jiangping/aspect_f1_card.png` | 路演 |

## 3.4 对接（属性）

| 交谁 | 交什么 |
| ---- | ------ |
| 属性 BERT（待定） | 同一 `aspect_sentiment.csv` 与划分说明 |
| 1号 / 6号 | 指标与样例图 |

------------------------------------------------------------------------

# 四、交付自检

| 检查项 | 达标 |
| ------ | ---- |
| 双向 BiLSTM+Attention 已实现 | ☐ |
| 情感 F1≥0.88 | ☐ |
| 相对 CNN F1 +≥2% | ☐ |
| Baseline/CNN/BiLSTM 三方对比完成 | ☐ |
| `bilstm_aspect_best.pt`；属性 F1≥85% | ☐ |
| 路演：三方对比图 + 属性 F1 卡 | ☐ |

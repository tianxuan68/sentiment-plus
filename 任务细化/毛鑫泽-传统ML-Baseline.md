# 毛鑫泽 · 传统 ML 情感 Baseline · 任务细化

> 对齐主设计：商品评论情感分类 · 传统 ML Baseline  
> 全局约定见：[00-总览与全局约定.md](./00-总览与全局约定.md)

------------------------------------------------------------------------

# 一、岗位速览

| 项 | 内容 |
| -- | ---- |
| **人员** | 毛鑫泽 |
| **主责** | 传统 ML 情感分类；正负预测；建 Baseline |
| **目录** | `sentiment-ai/pipelines/sentiment/baseline/` |
| **使用技术** | TF-IDF、KNN、Decision Tree、Random Forest、Scikit-learn |
| **验收标准** | ①≥3 种传统模型；②测试集 Acc≥85%；③输出 Acc/P/R/F1；④导出最佳 Baseline |
| **路演目录** | `sentiment-ai/pitch_assets/mao_xinze/` |

------------------------------------------------------------------------

# 二、输入

- `clean_train.csv` 的 `text_clean`、`label`（优先）；若暂无则用 `train.csv.sentence`
- **必须**读 `data/processed/splits/`（seed=68），禁止自划测试集

------------------------------------------------------------------------

# 三、实现步骤

1. 建议文件结构：
   ```
   pipelines/sentiment/baseline/
   ├── train_baseline.py
   ├── evaluate.py
   └── export_best.py
   ```
2. **仅在训练集**拟合 `TfidfVectorizer`：
   - `max_features=20000`
   - `ngram_range=(1, 2)`
3. 至少训练 3 个分类器：`KNeighborsClassifier`、`DecisionTreeClassifier`、`RandomForestClassifier`；各自网格或手工调参。
4. 在**同一测试集**上算 Acc / P / R / F1（`classification_report`，可 `average='binary'` 或按正类说明）。
5. Acc≥85% 的最佳模型：
   - 导出 `artifacts/sentiment/baseline/best_model.joblib`（含 vectorizer + clf）
   - 写出 `artifacts/metrics/baseline_metrics.json`（三种模型全表 + best 名）
6. 【路演】用三种模型 Acc/F1 画分组柱状图 + 指标表 PNG。

------------------------------------------------------------------------

# 四、产出文件

| 路径 | 说明 |
| ---- | ---- |
| `artifacts/sentiment/baseline/best_model.joblib` | 最佳 Baseline |
| `artifacts/metrics/baseline_metrics.json` | Acc/P/R/F1 |
| `pitch_assets/mao_xinze/baseline_bar.png` | 对比柱状图 |
| `pitch_assets/mao_xinze/baseline_table.png` | 指标表 |

------------------------------------------------------------------------

# 五、路演素材

- 核心效果页：KNN/DT/RF 指标 → 分组柱状图 + 指标表（宽 ≥1920）。

------------------------------------------------------------------------

# 六、对接

| 交谁 | 交什么 |
| ---- | ------ |
| 陈江平 / 胡潇潇 | `baseline_metrics.json`（作对比链起点） |
| 6号 | `pitch_assets/mao_xinze/` |

------------------------------------------------------------------------

------------------------------------------------------------------------

# 八、函数级接口契约

> 文件：`pipelines/sentiment/baseline/train_baseline.py`、`evaluate.py`、`export_best.py`

## 8.1 本岗必须实现的函数

```python
def load_corpus(
    csv_path: str = "data/processed/clean_train.csv",
    split_dir: str = "data/processed/splits/",
    split: Literal["train", "val", "test"] = "train",
) -> tuple[list[str], list[int]]:
    """读语料并按 splits 过滤。
    入参: clean_train 路径、划分目录、split 名
    返回: (texts, labels) — texts 用 text_clean 列
    上游: 刘攀 run_preprocess() 产出
    """

def fit_vectorizer(texts: list[str]) -> TfidfVectorizer:
    """仅在 train 集 fit。
    入参: 训练集 text_clean 列表
    返回: 已 fit 的 TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    """

def train_classifier(
    X, y,
    model_name: Literal["knn", "dt", "rf"],
    **kwargs,
) -> BaseEstimator:
    """训练单个传统模型。
    入参: 稀疏特征矩阵 X、标签 y、模型名
    返回: 已 fit 的分类器
    """

def evaluate(y_true: list[int], y_pred: list[int]) -> dict:
    """统一评估。
    入参: 真实/预测标签
    返回: {
        "acc": float,
        "precision": float,
        "recall": float,
        "f1": float,
        "model": str,
    }
    """

def predict(text: str, pipeline: Pipeline) -> dict:
    """单条推理（可选，供对比实验）。
    入参: 原始或 text_clean 字符串；含 vectorizer+clf 的 Pipeline
    返回: { "label": int, "prob": float, "model": "baseline_rf" }
    """

def export_best(
    pipeline: Pipeline,
    metrics: dict,
    out_model: str = "artifacts/sentiment/baseline/best_model.joblib",
    out_metrics: str = "artifacts/metrics/baseline_metrics.json",
) -> None:
    """导出最佳模型与指标 JSON。
    下游: 胡潇潇 build_model_compare() 读 baseline_metrics.json
    """
```

## 8.2 上下游

| 方向 | 对象 | 契约 |
| ---- | ---- | ---- |
| 上游 | 刘攀 | `clean_train.csv` + `splits/` |
| 下游 | 陈江平 / 胡潇潇 | `baseline_metrics.json` 中 `acc`、`f1` |
| 下游 | 6号 | 柱状图 PNG |

## 8.3 metrics JSON 字段

```json
{
  "models": [
    { "name": "knn", "acc": 0.86, "precision": 0.85, "recall": 0.84, "f1": 0.845 }
  ],
  "best_model": "rf",
  "best": { "acc": 0.872, "f1": 0.861 }
}
```

------------------------------------------------------------------------

# 九、交付自检

| 检查项 | 达标 |
| ------ | ---- |
| ≥3 种传统模型 | ☐ |
| 测试集 Acc≥85% | ☐ |
| Acc/P/R/F1 已输出 | ☐ |
| 最佳 Baseline 已导出 | ☐ |
| 路演柱状图+指标表已交 | ☐ |

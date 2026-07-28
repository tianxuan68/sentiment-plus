# 待定 · BERT 属性级情感分类 · 任务细化

> 对齐主设计：商品属性情感分析 · BERT 句对输入；权重交 1 号  
> 人员补全后：改文件名/子目录为人名即可，接口约定不变。  
> 全局约定见：[00-总览与全局约定.md](./00-总览与全局约定.md)  
> 函数对照：[01-函数接口对照表.md](./01-函数接口对照表.md)

------------------------------------------------------------------------

# 一、岗位速览

| 项 | 内容 |
| -- | ---- |
| **人员** | 待定 |
| **主责** | BERT 属性级情感 |
| **目录** | `pipelines/aspect_sentiment/bert/` |
| **使用技术** | `bert-base-chinese`、句对输入 |
| **验收标准** | ①Acc≥90%；②一句多属性预测；③权重交 1 号 |

------------------------------------------------------------------------

# 二、输入

| 来源 | 函数 / 文件 |
| ---- | ----------- |
| 5号 | `load_aspect_dataset()` ← `aspect_sentiment.csv` |

CSV 列契约：`sentence: str`, `aspect: str`, `polarity: int` (0|1)

------------------------------------------------------------------------

# 三、函数级接口契约

> 文件：`pipelines/aspect_sentiment/bert/train.py`、`predict.py`、`app/api/aspect.py`

```python
def load_aspect_dataset(
    csv_path: str = "data/processed/aspect_sentiment.csv",
    split_dir: str = "data/processed/splits_aspect/",
) -> Dataset:
    """上游: 5号 export_aspect_sentiment()"""

def encode_sentence_aspect(sentence: str, aspect: str, max_length: int = 128) -> dict:
    """句对编码。
    格式: [CLS] sentence [SEP] aspect [SEP]
    返回: { "input_ids", "attention_mask", "token_type_ids" }
    """

def train_bert_aspect(config: dict) -> dict:
    """微调二分类。
    返回: {
        "acc": float,                # ≥0.90
        "f1": float,
        "model_dir": "artifacts/aspect/bert_aspect_best/",
    }
    """

def predict_aspect(text: str, aspect: str) -> dict:
    """单句单属性。
    入参: text — 评论；aspect — 标准属性名
    返回: {
        "aspect": str,
        "polarity": int,
        "polarity_text": str,        # "正向"|"负向"
        "prob": float,
    }
    """

def predict_aspects(text: str, aspects: list[str]) -> list[dict]:
    """一句多属性（主推理入口）。
    入参:
        text: 评论文本
        aspects: 待测属性列表，如 ["质量","价格","物流","包装","服务"]
    返回: list[predict_aspect 返回结构]，顺序与 aspects 一致
    下游: 郑平高 _real_analyze() → AnalyzeResult.aspects[]
    """
```

### HTTP（sentiment-ai · 与陈江平 BiLSTM 属性共用路径）

| 方法 | 路径 | 请求 `body` | 响应 `result` |
| ---- | ---- | ----------- | ------------- |
| POST | `/api/v1/aspect/predict` | `{ "text": str, "aspects": list[str] }` | `{ "items": [{ "aspect", "polarity", "polarity_text", "prob" }] }` |

字段与郑平高 `AspectItem` 对齐：

| AI 返回 | schema 字段 | 类型 |
| ------- | ----------- | ---- |
| aspect | aspect | str |
| polarity | polarity | int |
| polarity_text | polarity_text | str |
| prob | prob | float |

### 郑平高侧消费

```python
# sentiment_ai_client._real_analyze()
aspect_raw = await _post_ai(client, "/api/v1/aspect/predict", {
    "text": text,
    "aspects": list(_ASPECT_HINTS),  # 质量,价格,物流,...
})
items = aspect_raw.get("items") or aspect_raw.get("aspects") or []
```

------------------------------------------------------------------------

# 四、产出文件

| 路径 | 说明 |
| ---- | ---- |
| `artifacts/aspect/bert_aspect_best/` | 权重 |
| `artifacts/metrics/bert_aspect_metrics.json` | acc/f1 |
| 多属性面板截图 | 交 6 号 / 1 号 |

------------------------------------------------------------------------

# 五、对接

| 交谁 | 交什么 |
| ---- | ------ |
| **郑平高（1号）** | `predict_aspects()` 或 POST `/api/v1/aspect/predict` |
| 陈江平 | 同一 `aspect_sentiment.csv` |
| 6号 | 面板截图 |

------------------------------------------------------------------------

# 六、交付自检

| 检查项 | 达标 |
| ------ | ---- |
| Acc≥90% | ☐ |
| `predict_aspects(text, aspects[])` 返回 items 长度 = len(aspects) | ☐ |
| 响应字段与 AspectItem 一致 | ☐ |
| 权重/HTTP 交 1 号 | ☐ |
| 路演面板截图已交 | ☐ |

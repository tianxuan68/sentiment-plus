# Sentiment-Plus

商品评论情感分析系统。管理端基于 Jeecg FastAPI + Vue3；AI 能力独立为 `sentiment-ai`，按团队分工分包，通过 HTTP 与后端协作。

设计与验收细节见：[商品评论情感分析项目设计2.md](./商品评论情感分析项目设计2.md)  
按人可执行任务细化见：[任务细化/](./任务细化/)（入口：[00-总览与全局约定.md](./任务细化/00-总览与全局约定.md)）

## 总体架构

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  jeecgboot-vue3 │────▶│   jeecg-fastapi  │────▶│  sentiment-ai   │
│  前端（郑平高）  │     │  业务编排 / 鉴权  │     │  模型训练与推理  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

| 路径 | 职责 | 主要负责人 |
|------|------|------------|
| `jeecgboot-vue3/` | 前端：系统管理、评论分析可视化（ECharts） | 郑平高（1号） |
| `jeecg-fastapi/` | 后端：鉴权、业务 API，转发/聚合 AI 接口 | 郑平高（1号） |
| `sentiment-ai/` | 数据、预处理、情感/关键词/属性模型与推理 | 见下表 |
| `商品评论情感分析项目设计2.md` | 分工、验收、路演素材规范 | 全组 |

## 功能模块 ↔ 人员 ↔ 目录

| 功能模块 | 人员 | 负责任务 | 代码/产物目录（规划） |
|----------|------|----------|----------------------|
| 商品评论情感分类 | 毛鑫泽 | 传统 ML Baseline（TF-IDF + ≥3 模型） | `sentiment-ai/pipelines/sentiment/baseline/` |
| | 陈江平 | BiLSTM+Attention；相对 CNN +2% F1 | `sentiment-ai/pipelines/sentiment/bilstm/` |
| | 胡潇潇 | BERT 微调；主推理入口交 1 号 | `sentiment-ai/pipelines/sentiment/bert/`、`app/` |
| PPT 制作 | 6号 | 路演 PPT 汇总成片 | `sentiment-ai/pitch_assets/` |
| 用户评价分析 | 郑平高 | 前后端 + 比例图/趋势图 | `jeecg-fastapi/`、`jeecgboot-vue3/` |
| | 杨国东 | 数据采集与 `unified_reviews.csv` | `sentiment-ai/data/` |
| | 待定 | 满意度统计（正负比/评分分布） | `sentiment-ai/analytics/` |
| 商品优缺点挖掘 | 邓新晓 | KeyBERT → pros/cons Top10 | `sentiment-ai/pipelines/pros_cons/` |
| 评论关键词提取 | 刘攀 | 清洗分词 → `clean_train.csv` | `sentiment-ai/pipelines/preprocess/` |
| | 杨国东 | KeyBERT 关键词 + TF-IDF 对照 | `sentiment-ai/pipelines/keywords/` |
| 商品属性情感分析 | 5号 | BIO+polarity；NER | `sentiment-ai/pipelines/aspect_ner/` |
| | 陈江平 | BiLSTM 属性情感 | `sentiment-ai/pipelines/aspect_sentiment/bilstm/` |
| | 待定 | BERT 属性级情感 | `sentiment-ai/pipelines/aspect_sentiment/bert/` |

**数据约定（摘要）**

- 中文：`train.csv` → `sentence`、`label`、`dataset`
- 英文：`data.csv` → `reviewText`、`overall`、`asin`（≥5 万）
- 划分：train:val:test = 8:1:1，`random_seed=68`，全组共用；中英不混训
- 情感对比链：Baseline → TextCNN → BiLSTM → BERT（四模型）

## 仓库目录结构

```
sentiment-plus/
├── jeecg-fastapi/                    # 后端（郑平高）
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── db/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── utils/
│   ├── scripts/
│   └── sql/
├── jeecgboot-vue3/                   # 前端（郑平高）
│   └── src/
├── sentiment-ai/                     # AI 模块（按分工分包）
├── 商品评论情感分析项目设计2.md
├── README.md
└── LICENSE
```

## AI 模块目录（`sentiment-ai/`）

```
sentiment-ai/
├── app/                              # 推理 HTTP 服务（对接郑平高；主入口胡潇潇）
│   ├── api/
│   ├── core/
│   ├── schemas/
│   ├── services/
│   └── utils/
├── pipelines/
│   ├── preprocess/                   # 刘攀：清洗 / jieba / 停用词
│   ├── sentiment/
│   │   ├── baseline/                 # 毛鑫泽：TF-IDF + KNN/DT/RF
│   │   ├── cnn/                      # TextCNN（对比实验用）
│   │   ├── bilstm/                   # 陈江平：Embedding+BiLSTM+Attention
│   │   └── bert/                     # 胡潇潇：BertForSequenceClassification
│   ├── keywords/                     # 杨国东：KeyBERT vs TF-IDF
│   ├── pros_cons/                    # 邓新晓：正/负桶 Top10
│   ├── aspect_ner/                   # 5号：BIO NER + polarity 标注产物
│   └── aspect_sentiment/
│       ├── bilstm/                   # 陈江平：bilstm_aspect_best.pt
│       └── bert/                     # 待定：句对属性情感
├── analytics/                        # 待定：正负比例 / 评分分布 / 满意度报告
├── ml/                               # 共用模型组件（可选复用）
│   ├── models/
│   ├── trainers/
│   └── evaluators/
├── configs/                          # 划分种子、超参等
├── data/
│   ├── raw/                          # train.csv / data.csv
│   ├── processed/                    # unified_reviews / clean_train 等
│   └── samples/
├── artifacts/                        # 权重与导出指标（gitignore 大文件）
├── pitch_assets/                     # 路演图：各员子目录 → 6号汇总
├── scripts/
├── tests/
└── docs/
```
**.gitkeep文件为占位文件，可以删除**
**分层约定**

- `pipelines/*`：按功能与负责人隔离，训练脚本与产出放本目录
- `app/`：只暴露推理/健康检查，不写训练细节
- `data/` + `artifacts/`：原始数据与权重分离；大文件不入库
- `pitch_assets/<人员>/`：路演 PNG/表，交 6 号做 PPT

## 快速开始（管理端）

### 后端

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
cd jeecg-fastapi
copy .env.example .env
..\.venv\Scripts\python scripts/init_slim_db.py
..\.venv\Scripts\python run.py
```

- API 前缀：`http://localhost:8000/jeecg-boot`
- 默认账号：`admin` / `123456`

### 前端

```powershell
cd jeecgboot-vue3
pnpm install
pnpm dev
```

开发服务器默认：`http://localhost:3100`（代理到后端 `8000`）。

## 功能入口（当前管理壳）

登录后默认进入用户管理：`/system/user`。

- 用户管理 `/system/user`
- 角色管理 `/system/role`
- 菜单管理 `/system/menu`
- 部门管理 `/system/depart`
- 数据字典 `/system/dict`

> 评论分析业务页与 AI 推理接入由郑平高按设计文档接口约定推进；`sentiment-ai` 当前以目录与文档规划为主。

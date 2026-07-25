# Sentiment-Plus

商品评论情感分析系统：基于 Jeecg FastAPI + Vue3，提供情感分类、关键词提取与属性级情感分析。

## 目录结构

| 路径 | 说明 |
|------|------|
| `jeecg-fastapi/` | 后端（FastAPI） |
| `jeecgboot-vue3/` | 前端（Vue3 + Ant Design Vue） |
| `docs/` | 项目设计文档 |

## 快速开始

### 后端

```powershell
cd jeecg-fastapi
copy .env.example .env
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
python scripts/init_slim_db.py
.\.venv\Scripts\python run.py
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

## 业务入口

登录后默认进入情感分析看板：`/sentiment/dashboard`。

主要接口（需登录）：

- `GET  /sys/sentimentAnalysis/overview`
- `POST /sys/sentimentAnalysis/predict`
- `POST /sys/sentimentAnalysis/predictBatch`
- `POST /sys/sentimentAnalysis/keywords`
- `POST /sys/sentimentAnalysis/aspects`

当前分析引擎为词典基线（baseline）；深度学习模型接口已预留，可按 `docs/项目设计.md` 迭代接入。

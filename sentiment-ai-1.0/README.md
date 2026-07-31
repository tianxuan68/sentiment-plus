# sentiment-ai 1.0 · 好评 / 差评（二分类）

> 目录：`sentiment-ai/`（等同链接 `sentiment-ai-1.0/`）  
> 版本：见 `VERSION` → **1.0.0**  
> 下一代：[`../sentiment-ai-2.0/`](../sentiment-ai-2.0/)（多标签 / 动态标签，开发中）

## 1.0 做什么

| 能力 | 说明 |
|------|------|
| 整句情感 | `label`：**0=差评，1=好评** |
| 关键词 / 优缺点 / 属性 | 仍挂在本服务（或同仓库 pipelines） |
| 数据主集 | `train2.csv`（中文 0/1） |

**不是**开心/伤心等多情绪分类；那是 2.0 规划方向之一。

## 启动

```powershell
cd sentiment-ai   # 或 sentiment-ai-1.0
# 依赖在仓库根目录：pip install -r ../requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8100 --reload
```

业务后端默认：`SENTIMENT_AI_BASE_URL=http://127.0.0.1:8100`

## 核心接口（1.0）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/sentiment/predict` | 二分类推理 |

默认请求体示例：`{"texts": ["物流很快"]}` → 每条含 `label` / `label_name` / `confidence`。

## 与 2.0 关系

```
1.0（本目录）  稳定：好评/差评 + 现有 pipelines
2.0（sentiment-ai-2.0）  新开：固定多标签 → 再做「拼多多式」按商品动态标签
```

郑平高当前联调 **只接 1.0（8100）**；2.0 就绪后再加配置项切换。

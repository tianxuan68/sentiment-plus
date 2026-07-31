# sentiment-ai 2.0 · 多标签 / 动态标签（开发中）

> 与 [`sentiment-ai` 1.0](../sentiment-ai/) **并列**，互不覆盖。  
> 1.0 = 好评/差评二分类（已稳定，端口 **8100**）  
> 2.0 = 多标签多分类 → 再做「拼多多式」按商品动态标签（端口建议 **8200**）

## 版本路线

| 阶段 | 目标 | 状态 |
|------|------|------|
| **2.0-A** | 固定标签集的多分类 / 多标签（如：物流快、质量好、包装差…） | 骨架已建，待实现 |
| **2.0-B** | 动态：不同商品/类目，标签体系不同（拼多多风格） | **等组长安排再做** |

## 和 1.0 怎么配合（推荐）

```
评论
  ├─ 1.0：先出 好评/差评（粗筛）
  └─ 2.0：再出 多标签 / 动态标签（细挖）
```

业务后端以后可同时调两个服务，或只切 2.0；**当前郑平高仍只接 1.0**。

## 目录

```
sentiment-ai-2.0/
├── VERSION                 # 2.0.0-dev
├── README.md
├── requirements.txt        # 指向仓库根目录统一依赖
├── app/                    # FastAPI 推理（8200）
│   ├── main.py
│   ├── api/
│   ├── schemas/
│   └── services/
├── pipelines/
│   ├── multilabel/         # 2.0-A：固定多标签
│   └── dynamic_tags/       # 2.0-B：按商品动态标签（预留）
├── configs/
├── data/                   # 2.0 自用数据（可软链/复用 1.0 raw）
├── artifacts/
├── docs/
│   └── ROADMAP.md
└── tests/
```

## 启动（骨架可跑，接口为占位）

```powershell
cd sentiment-ai-2.0
# pip install -r ../requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8200 --reload
```

打开：http://127.0.0.1:8200/docs

## 计划接口（实现前仅占位）

| 阶段 | 方法 | 路径 | 说明 |
|------|------|------|------|
| A | POST | `/api/v2/tags/predict` | 固定标签多分类/多标签 |
| B | POST | `/api/v2/tags/predict_dynamic` | 按 `product_id`/`category` 动态标签 |
| — | GET | `/health` | 健康检查 |

字段契约等 2.0-A 开工时再钉死，避免和 1.0 的 `label:0|1` 混用。

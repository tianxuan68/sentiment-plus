# Sentiment-Plus（系统配置精简版）

基于 Jeecg FastAPI + Vue3 的后台管理系统壳，仅保留系统配置相关能力：用户、角色、菜单、部门、数据字典与登录鉴权。

## 目录结构

| 路径 | 说明 |
|------|------|
| `jeecg-fastapi/` | 后端（FastAPI） |
| `jeecgboot-vue3/` | 前端（Vue3 + Ant Design Vue） |

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

## 功能入口

登录后默认进入用户管理：`/system/user`。

系统管理菜单：

- 用户管理 `/system/user`
- 角色管理 `/system/role`
- 菜单管理 `/system/menu`
- 部门管理 `/system/depart`
- 数据字典 `/system/dict`

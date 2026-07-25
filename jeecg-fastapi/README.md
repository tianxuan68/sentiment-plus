# Sentiment-Plus 后端

JeecgBoot 精简版 FastAPI：系统配置 API + 评论情感分析。

```powershell
copy .env.example .env
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
python scripts/init_slim_db.py
.\.venv\Scripts\python run.py
```

API：`http://localhost:8000/jeecg-boot`  
默认账号：`admin` / `123456`

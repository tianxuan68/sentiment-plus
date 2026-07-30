"""sentiment-ai FastAPI 入口（推理服务，对接 1 号）。"""

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="sentiment-ai", version="0.1.0")
app.include_router(router)


@app.get("/")
def root():
    return {"service": "sentiment-ai", "docs": "/docs"}

"""sentiment-ai 2.0 入口（多标签 / 动态标签，开发中）。"""

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="sentiment-ai-2.0",
    version="2.0.0-dev",
    description="多标签与动态标签。稳定二分类请用 sentiment-ai 1.0（8100）。",
)
app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "sentiment-ai-2.0",
        "version": "2.0.0-dev",
        "mode": "multilabel_scaffold",
        "port_suggest": 8200,
        "stable_binary": "sentiment-ai 1.0 @ 8100",
        "docs": "/docs",
    }

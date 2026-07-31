"""sentiment-ai 1.0 FastAPI 入口（好评/差评二分类，对接 1 号）。"""

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="sentiment-ai-1.0",
    version="1.0.0",
    description="二分类情感：0=差评，1=好评。多标签见 sentiment-ai-2.0。",
)
app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "sentiment-ai",
        "version": "1.0.0",
        "mode": "binary_sentiment",
        "docs": "/docs",
        "next": "sentiment-ai-2.0",
    }

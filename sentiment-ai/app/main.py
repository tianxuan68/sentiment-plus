"""
sentiment-ai FastAPI 入口（推理服务，对接 1 号）
"""

# 1.导包
from fastapi import FastAPI

from app.api.routes import router

# 2.创建应用
app = FastAPI(title='sentiment-ai', version='0.1.0')
app.include_router(router)


@app.get('/')
def root():
    return {'service': 'sentiment-ai', 'docs': '/docs'}


if __name__ == '__main__':
    import uvicorn
    print('启动 sentiment-ai 推理服务...')
    uvicorn.run(app, host='0.0.0.0', port=8000)

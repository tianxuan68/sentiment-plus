"""
LLM 模型预测函数封装
FastAPI 框架，绑定到 /predict

步骤：
1. 导包
2. 创建 FastAPI 实例
3. 定义请求体模型（Pydantic）
4. 注册路由并实现视图函数
5. 启动服务（使用 uvicorn）
"""

import jieba
import pandas as pd
import pickle
import warnings
from bilstm_predict_fun import predict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# 2. 创建 FastAPI 实例
app = FastAPI(title="评论情感预测API", version="1.0")

# 3. 定义请求体数据模型
class PredictRequest(BaseModel):
    text: str  # 新闻标题

# 4. 注册路由
@app.post("/predict")
async def llm_predict(request: PredictRequest):
    """
    1. 接收请求参数（自动校验）
    2. 执行预测函数
    3. 返回结果
    """
    # 打印请求内容（便于调试）
    print(request.dict())

    # 因为 Pydantic 已确保 text 字段存在且为 str，无需手动校验
    # 但若需要额外校验，可以在这里添加

    # 调用预测函数，传入字典或直接传入 text
    # 假设 predict 函数接收字典 {"text": ...} 或直接接收字符串，根据原代码推断它接收整个 input_request（字典）
    # 原代码：result = predict(input_request)，其中 input_request 是 {"text": "..."}
    # 所以我们传入 request.dict()
    result = predict(request.dict())

    # FastAPI 会自动将返回的 dict 转为 JSON，无需 json.dumps
    return result

# 5. 启动服务（直接运行该脚本时）
# 5. 启动服务（直接运行该脚本时）
if __name__ == "__main__":
    import os

    # 动态获取当前文件名（不含 .py 后缀）
    module_name = os.path.splitext(os.path.basename(__file__))[0]

    uvicorn.run(
        f"{module_name}:app",  # ✅ 改为动态获取！
        host="127.0.0.1",
        port=8000,
        reload=True
    )
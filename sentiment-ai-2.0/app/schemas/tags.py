from pydantic import BaseModel, Field


class TagPredictRequest(BaseModel):
    """2.0-A：固定标签多分类/多标签。"""

    text: str = Field(..., min_length=1, description="评论文本")
    top_k: int = Field(5, ge=1, le=50, description="返回前 K 个标签")


class DynamicTagPredictRequest(BaseModel):
    """2.0-B：动态标签（拼多多风格预留）。"""

    text: str = Field(..., min_length=1, description="评论文本")
    product_id: str | None = Field(None, description="商品 ID")
    category_id: str | None = Field(None, description="类目 ID")
    top_k: int = Field(5, ge=1, le=50)


class TagItem(BaseModel):
    tag: str
    score: float
    polarity: str | None = Field(
        None, description="可选：positive/negative，与标签搭配"
    )


class TagPredictResponse(BaseModel):
    text: str
    tags: list[TagItem]
    implemented: bool = Field(
        False, description="False 表示仍为占位，勿当生产结果"
    )
    message: str = ""

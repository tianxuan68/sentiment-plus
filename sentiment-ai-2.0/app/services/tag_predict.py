"""2.0 标签预测服务（占位）。真实模型后续接到 pipelines/multilabel 与 dynamic_tags。"""

from app.schemas.tags import (
    DynamicTagPredictRequest,
    TagItem,
    TagPredictRequest,
    TagPredictResponse,
)


def predict_fixed_tags(body: TagPredictRequest) -> TagPredictResponse:
    return TagPredictResponse(
        text=body.text,
        tags=[],
        implemented=False,
        message="2.0-A 未实现：请先完成固定多标签数据与模型；二分类请调 1.0 /api/sentiment/predict",
    )


def predict_dynamic_tags(body: DynamicTagPredictRequest) -> TagPredictResponse:
    return TagPredictResponse(
        text=body.text,
        tags=[],
        implemented=False,
        message=(
            "2.0-B 未实现：动态标签（按 product_id/category_id）等安排后再做；"
            f"收到 product_id={body.product_id!r}, category_id={body.category_id!r}"
        ),
    )

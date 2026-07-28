"""5号 · 商品属性实体识别（BIO NER + polarity）。"""

from pipelines.aspect_ner.annotate import annotate_bio, build_annotations_from_pool, load_annotation_pool
from pipelines.aspect_ner.export import export_aspect_sentiment, export_ner_jsonl
from pipelines.aspect_ner.predict import predict_entities

__all__ = [
    "load_annotation_pool",
    "annotate_bio",
    "build_annotations_from_pool",
    "export_ner_jsonl",
    "export_aspect_sentiment",
    "predict_entities",
]

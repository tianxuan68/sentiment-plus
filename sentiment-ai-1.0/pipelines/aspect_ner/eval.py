"""实体级 F1 评估（seqeval 或手写实现）。"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from pipelines.aspect_ner.constants import aspect_from_bio


def _to_entities(labels: list[str]) -> set[tuple[int, int, str]]:
    """BIO 序列 -> {(start, end, type), ...} 词级 span。"""
    entities: list[tuple[str, int, int]] = []
    start, ent_type = -1, None
    for i, lab in enumerate(labels):
        if lab.startswith("B-"):
            if start >= 0 and ent_type:
                entities.append((ent_type, start, i - 1))
            ent_type = aspect_from_bio(lab)
            start = i
        elif lab.startswith("I-") and ent_type == aspect_from_bio(lab):
            continue
        else:
            if start >= 0 and ent_type:
                entities.append((ent_type, start, i - 1))
            start, ent_type = -1, None
    if start >= 0 and ent_type:
        entities.append((ent_type, start, len(labels) - 1))
    return {(t, s, e) for t, s, e in entities}


def entity_f1(y_true: Iterable[list[str]], y_pred: Iterable[list[str]]) -> dict:
    tp = fp = fn = 0
    for t_labels, p_labels in zip(y_true, y_pred):
        t_set = _to_entities(t_labels)
        p_set = _to_entities(p_labels)
        tp += len(t_set & p_set)
        fp += len(p_set - t_set)
        fn += len(t_set - p_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def aspect_coverage(sentiment_records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in sentiment_records:
        counts[str(row.get("aspect", ""))] += 1
    return dict(counts)

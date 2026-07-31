"""
实体级 F1 评估（手写 BIO 实体匹配）
"""

#  评估核心: 斌子法则 1212
# 1.导包
from collections import defaultdict

from pipelines.aspect_ner.constants import aspect_from_bio


def _to_entities(labels):
    # BIO 序列 -> {(type, start, end), ...}
    entities = []
    start, ent_type = -1, None
    for i, lab in enumerate(labels):
        if lab.startswith('B-'):
            if start >= 0 and ent_type:
                entities.append((ent_type, start, i - 1))
            ent_type = aspect_from_bio(lab)
            start = i
        elif lab.startswith('I-') and ent_type == aspect_from_bio(lab):
            continue
        else:
            if start >= 0 and ent_type:
                entities.append((ent_type, start, i - 1))
            start, ent_type = -1, None
    if start >= 0 and ent_type:
        entities.append((ent_type, start, len(labels) - 1))
    return {(t, s, e) for t, s, e in entities}


def entity_f1(y_true, y_pred):
    # 实体级 precision / recall / f1
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
    result = {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }
    print(f'实体 F1：{result["f1"]}，精确率：{result["precision"]}，召回率：{result["recall"]}')
    return result


def aspect_coverage(sentiment_records):
    # 各属性出现次数
    counts = defaultdict(int)
    for row in sentiment_records:
        counts[str(row.get('aspect', ''))] += 1
    return dict(counts)


if __name__ == '__main__':
    y_true = [['O', 'B-物流', 'I-物流', 'O']]
    y_pred = [['O', 'B-物流', 'I-物流', 'O']]
    print(entity_f1(y_true, y_pred))

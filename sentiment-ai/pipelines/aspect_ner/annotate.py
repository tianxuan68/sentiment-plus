"""
BIO + polarity 双标注：读抽样池、半自动标注
"""

# 1.导包
import csv
from pathlib import Path

import jieba

from pipelines.aspect_ner.constants import (
    ASPECT_KEYWORDS,
    ASPECT_TYPES,
    BIO_LABEL_O,
    NEG_HINTS,
    POS_HINTS,
    polarity_text,
)


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(Path(__file__).resolve().parents[2]).replace('\\', '/') + '/'
        self.pool_path = self.root_path + 'data/samples/aspect_annotate_pool.csv'


config = Config()


def load_annotation_pool(path=None):
    # 3.加载标注池（上游：杨国东抽样）
    csv_path = Path(path) if path else Path(config.pool_path)
    if not csv_path.is_absolute():
        csv_path = Path(config.root_path) / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f'标注池不存在：{csv_path}，请先运行 scripts/build_aspect_annotate_pool.py')

    rows = []
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = (row.get('sentence') or '').strip()
            if not sentence:
                continue
            rows.append({'id': str(row.get('id', len(rows))), 'sentence': sentence})
    print(f'标注池条数：{len(rows)}')
    return rows


def find_aspect_spans(sentence):
    # 在句中找属性触发词 span -> [(start, end, aspect_type), ...]
    spans = []
    seen = set()
    for aspect in ASPECT_TYPES:
        for kw in ASPECT_KEYWORDS.get(aspect, (aspect,)):
            start = 0
            while True:
                idx = sentence.find(kw, start)
                if idx < 0:
                    break
                span = (idx, idx + len(kw))
                if span not in seen:
                    seen.add(span)
                    spans.append((idx, idx + len(kw), aspect))
                start = idx + 1
    return sorted(spans, key=lambda x: x[0])


def infer_polarity(sentence, aspect, sentence_label=None):
    # 方面极性：局部上下文 + 句级 label 兜底
    window = sentence
    for kw in ASPECT_KEYWORDS.get(aspect, (aspect,)):
        pos = sentence.find(kw)
        if pos >= 0:
            window = sentence[max(0, pos - 8): pos + len(kw) + 8]
            break
    if any(h in window for h in NEG_HINTS):
        return 0
    if any(h in window for h in POS_HINTS):
        return 1
    if sentence_label is not None:
        return int(sentence_label)
    return 1


def tokenize_sentence(sentence):
    return [t for t in jieba.lcut(sentence.strip()) if t.strip()]


def spans_to_token_labels(tokens, spans, sentence):
    # 字符 span 对齐到 jieba 词级 BIO
    labels = [BIO_LABEL_O] * len(tokens)
    if not tokens:
        return labels

    cursor = 0
    token_spans = []
    for tok in tokens:
        idx = sentence.find(tok, cursor)
        if idx < 0:
            idx = cursor
        token_spans.append((idx, idx + len(tok)))
        cursor = idx + len(tok)

    for start, end, aspect in spans:
        first = True
        for i, (ts, te) in enumerate(token_spans):
            if te <= start or ts >= end:
                continue
            labels[i] = f'B-{aspect}' if first else f'I-{aspect}'
            first = False
    return labels


def annotate_bio(record_id, sentence, aspect_spans=None, sentence_label=None):
    # 单句 BIO 标注 + 方面情感行
    spans = aspect_spans if aspect_spans is not None else find_aspect_spans(sentence)
    tokens = tokenize_sentence(sentence)
    labels = spans_to_token_labels(tokens, spans, sentence)

    sentiment_rows = []
    aspects_in_span = set()
    for _s, _e, aspect in spans:
        if aspect in aspects_in_span:
            continue
        aspects_in_span.add(aspect)
        pol = infer_polarity(sentence, aspect, sentence_label)
        sentiment_rows.append({
            'id': record_id,
            'sentence': sentence,
            'aspect': aspect,
            'polarity': pol,
            'polarity_text': polarity_text(pol),
            'aspect_span': f'{_s}:{_e}',
        })

    return {
        'id': record_id,
        'sentence': sentence,
        'tokens': tokens,
        'labels': labels,
        'sentiment_rows': sentiment_rows,
    }


def build_annotations_from_pool(pool, labels_by_id=None):
    # 批量标注 -> (ner_records, sentiment_records)
    ner_records = []
    sentiment_records = []
    labels_by_id = labels_by_id or {}

    for item in pool:
        rid = str(item['id'])
        sentence = item['sentence']
        rec = annotate_bio(rid, sentence, sentence_label=labels_by_id.get(rid))
        ner_records.append({
            'id': rec['id'],
            'tokens': rec['tokens'],
            'labels': rec['labels'],
            'sentence': rec['sentence'],
        })
        sentiment_records.extend(rec['sentiment_rows'])

    print(f'NER 记录：{len(ner_records)}，情感行：{len(sentiment_records)}')
    return ner_records, sentiment_records


if __name__ == '__main__':
    demo = annotate_bio('1', '物流很快，包装简陋', sentence_label=1)
    print(f'tokens：{demo["tokens"]}')
    print(f'labels：{demo["labels"]}')
    print(f'sentiment_rows：{demo["sentiment_rows"]}')

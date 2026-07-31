"""
属性 NER 推理：predict_entities / predict_fun
供郑平高 AnalyzeResult.entities 对齐
"""

# 1.导包
import json
from pathlib import Path

from pipelines.aspect_ner.annotate import find_aspect_spans, infer_polarity
from pipelines.aspect_ner.constants import ASPECT_TYPES, polarity_text


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(Path(__file__).resolve().parents[2]).replace('\\', '/') + '/'
        self.model_dir = self.root_path + 'artifacts/aspect_ner/best'
        self.metrics_path = self.root_path + 'artifacts/metrics/ner_f1.json'
        self.max_length = 128
        self.prefer_model = True


config = Config()


def _rule_predict(text):
    # 规则基线：关键词 span + 极性
    sentence = text.strip()
    if not sentence:
        return []
    spans = find_aspect_spans(sentence)
    entities = []
    seen = set()
    for start, end, aspect in spans:
        if aspect in seen:
            continue
        seen.add(aspect)
        pol = infer_polarity(sentence, aspect)
        entities.append({
            'aspect': aspect,
            'text': sentence[start:end],
            'start': start,
            'end': end,
            'polarity': pol,
            'polarity_text': polarity_text(pol),
        })
    if not entities:
        for aspect in ASPECT_TYPES:
            if aspect in sentence:
                idx = sentence.find(aspect)
                pol = infer_polarity(sentence, aspect)
                entities.append({
                    'aspect': aspect,
                    'text': aspect,
                    'start': idx,
                    'end': idx + len(aspect),
                    'polarity': pol,
                    'polarity_text': polarity_text(pol),
                })
                break
    return sorted(entities, key=lambda e: e['start'])


def _model_predict(text):
    # 若已训练 BERT NER 权重则加载推理；否则返回 None
    model_dir = Path(config.model_dir)
    if not (model_dir / 'config.json').exists():
        return None
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError:
        return None

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    my_model = AutoModelForTokenClassification.from_pretrained(model_dir)
    my_model.eval()

    tokens = list(text.strip())
    if not tokens:
        return []

    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors='pt',
        truncation=True,
        max_length=config.max_length,
    )
    with torch.no_grad():
        logits = my_model(**enc).logits[0]
    pred_ids = logits.argmax(-1).tolist()
    id2label = my_model.config.id2label

    word_ids = enc.word_ids()
    labels = []
    last_wid = None
    for i, wid in enumerate(word_ids):
        if wid is None or wid == last_wid:
            continue
        last_wid = wid
        if wid < len(tokens):
            labels.append(id2label.get(str(pred_ids[i]), 'O'))

    entities = []
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab.startswith('B-'):
            aspect = lab[2:]
            start = i
            i += 1
            while i < len(labels) and labels[i] == f'I-{aspect}':
                i += 1
            end = i
            pol = infer_polarity(text, aspect)
            entities.append({
                'aspect': aspect,
                'text': text[start:end],
                'start': start,
                'end': end,
                'polarity': pol,
                'polarity_text': polarity_text(pol),
            })
        else:
            i += 1
    return entities


def predict_entities(text, prefer_model=None):
    # 属性 NER 推理入口
    use_model = config.prefer_model if prefer_model is None else prefer_model
    if use_model:
        model_out = _model_predict(text)
        if model_out is not None:
            return model_out
    return _rule_predict(text)


def predict_fun(data):
    """
    推理 API（szai6 契约优先）:
      - dict: {'text': '...'} -> 写回 entities / predict_class
      - str / list[str]: 兼容旧调用，返回 {'results': [...]}
    """
    #  1.dict 契约
    if isinstance(data, dict):
        text = str(data.get('text', ''))
        ents = predict_entities(text)
        data['entities'] = ents
        data['predict_class'] = ents
        return data

    #  2.兼容 str / list
    texts = [data] if isinstance(data, str) else list(data)
    all_entities = []
    for text in texts:
        ents = predict_entities(text)
        all_entities.append({'text': text, 'entities': ents, 'predict_class': ents})
    return {'results': all_entities}


def load_metrics():
    path = Path(config.metrics_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    sample = {'text': '物流很快，包装简陋'}
    print(predict_fun(sample))
    print('-----------------')
    out = predict_fun(sample['text'])
    print(f'输入：{sample["text"]}')
    print(f'预测实体：{out["results"][0]["entities"]}')

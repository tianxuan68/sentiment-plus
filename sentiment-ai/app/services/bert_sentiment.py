"""
加载 artifacts/bert/best 做情感推理（训练完成后可用）
对齐 szai6：predict_fun(data)->dict，写入 predict_class
"""

# 1.导包
from functools import lru_cache
from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizer


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(Path(__file__).resolve().parents[2]).replace('\\', '/') + '/'
        self.model_dir = self.root_path + 'artifacts/bert/best'
        self.max_length = 128
        self.id2class = {0: '负向', 1: '正向'}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = Config()


@lru_cache(maxsize=1)
def _load(model_dir):
    # 3.加载模型（缓存）
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(
            f'未找到 BERT 权重：{path}。请先完成微调并导出到 artifacts/bert/best'
        )
    tokenizer = BertTokenizer.from_pretrained(path)
    my_model = BertForSequenceClassification.from_pretrained(path).to(config.device)
    my_model.eval()
    print(f'已加载情感模型：{path}，设备：{config.device}')
    return tokenizer, my_model, config.device


def _predict_one(text, tokenizer, my_model, device, max_len):
    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors='pt',
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = my_model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)
        pred = int(logits.argmax(dim=-1).item())
        conf = float(probs.max(dim=-1).values.item())
    return pred, conf


def predict_fun(data, model_dir=None, max_length=None):
    """
    推理 API（szai6 契约优先）:
      - dict: {'text': '...'} -> 写回 predict_class
      - str / list[str]: 返回 {'results': [...]}
    """
    tokenizer, my_model, device = _load(str(model_dir or config.model_dir))
    max_len = max_length or config.max_length

    if isinstance(data, dict):
        text = str(data.get('text', ''))
        pred, conf = _predict_one(text, tokenizer, my_model, device, max_len)
        data['label'] = pred
        data['label_name'] = config.id2class.get(pred, str(pred))
        data['confidence'] = conf
        data['predict_class'] = config.id2class.get(pred, str(pred))
        return data

    texts = [data] if isinstance(data, str) else list(data)
    results = []
    for text in texts:
        pred, conf = _predict_one(text, tokenizer, my_model, device, max_len)
        results.append({
            'text': text,
            'label': pred,
            'label_name': config.id2class.get(pred, str(pred)),
            'confidence': conf,
            'predict_class': config.id2class.get(pred, str(pred)),
        })
    return {'results': results}


def predict_sentiment(texts, model_dir=None, max_length=128):
    # 兼容 FastAPI 路由：返回 list[dict]，字段与 PredictItem 一致
    out = predict_fun(texts, model_dir=model_dir, max_length=max_length)
    return [
        {
            'text': item['text'],
            'label': item['label'],
            'label_name': item['label_name'],
            'confidence': item['confidence'],
        }
        for item in out['results']
    ]


if __name__ == '__main__':
    sample = {'text': '味道很好，会回购'}
    print(predict_fun(sample))
    print('-----------------')
    demo = ['味道很好，会回购', '物流太慢，很失望']
    out = predict_fun(demo)
    for item in out['results']:
        print(f'文本：{item["text"]}，预测：{item["predict_class"]}，置信度：{item["confidence"]:.4f}')

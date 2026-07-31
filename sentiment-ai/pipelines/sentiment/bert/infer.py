"""
BERT 本地推理自测（加载 artifacts/bert/best）
对齐 szai6：predict_fun(data)->dict，写入 predict_class
"""

# 1.导包
from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizer


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(Path(__file__).resolve().parents[3]).replace('\\', '/') + '/'
        self.model_dir = self.root_path + 'artifacts/bert/best'
        self.max_length = 128
        self.demo_texts = ['味道很好，会回购', '物流太慢，很失望']
        self.id2class = {0: '负向', 1: '正向'}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = Config()


def _predict_one(text, tokenizer, my_model):
    # 单条推理
    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=config.max_length,
        return_tensors='pt',
    )
    encoded = {k: v.to(config.device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = my_model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)
        pred = int(logits.argmax(dim=-1).item())
        conf = float(probs.max(dim=-1).values.item())
    return pred, conf


def predict_fun(data):
    """
    推理 API（szai6 契约优先）:
      - dict: {'text': '...'} -> 写回 predict_class 后返回
      - str / list[str]: 兼容旧调用，返回 {'results': [...]}
    """
    model_dir = Path(config.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f'未找到权重：{model_dir}')

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    my_model = BertForSequenceClassification.from_pretrained(model_dir).to(config.device)
    my_model.eval()

    #  1.dict 契约（课堂标准）
    if isinstance(data, dict):
        text = str(data.get('text', ''))
        pred, conf = _predict_one(text, tokenizer, my_model)
        data['label'] = pred
        data['label_name'] = config.id2class.get(pred, str(pred))
        data['confidence'] = conf
        data['predict_class'] = config.id2class.get(pred, str(pred))
        return data

    #  2.兼容 str / list
    texts = [data] if isinstance(data, str) else list(data)
    results = []
    for text in texts:
        pred, conf = _predict_one(text, tokenizer, my_model)
        results.append({
            'text': text,
            'label': pred,
            'label_name': config.id2class.get(pred, str(pred)),
            'confidence': conf,
            'predict_class': config.id2class.get(pred, str(pred)),
        })
    return {'results': results}


if __name__ == '__main__':
    # demo01：szai6 单条 dict
    sample = {'text': config.demo_texts[0]}
    print(predict_fun(sample))
    print('-----------------')
    # demo02：批量 list
    out = predict_fun(config.demo_texts)
    for item in out['results']:
        print(f'文本：{item["text"]}，预测：{item["predict_class"]}，置信度：{item["confidence"]:.4f}')

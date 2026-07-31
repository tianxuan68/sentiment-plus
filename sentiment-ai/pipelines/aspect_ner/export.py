"""
写出 aspect_ner.jsonl 与 aspect_sentiment.csv
"""

# 1.导包
import csv
import json
from pathlib import Path


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(Path(__file__).resolve().parents[2]).replace('\\', '/') + '/'
        self.ner_jsonl = self.root_path + 'data/processed/aspect_ner.jsonl'
        self.sentiment_csv = self.root_path + 'data/processed/aspect_sentiment.csv'


config = Config()


def export_ner_jsonl(records, path=None):
    # 导出 NER jsonl
    out = Path(path) if path else Path(config.ner_jsonl)
    if not out.is_absolute():
        out = Path(config.root_path) / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for rec in records:
            line = {
                'id': rec['id'],
                'tokens': rec['tokens'],
                'labels': rec['labels'],
            }
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(f'已导出 NER：{out}，条数={len(records)}')
    return len(records)


def export_aspect_sentiment(records, path=None):
    # 导出属性情感 csv
    out = Path(path) if path else Path(config.sentiment_csv)
    if not out.is_absolute():
        out = Path(config.root_path) / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['id', 'sentence', 'aspect', 'polarity', 'polarity_text', 'aspect_span']
    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, '') for k in fieldnames})
    print(f'已导出情感表：{out}，行数={len(records)}')
    return len(records)


if __name__ == '__main__':
    print(f'默认 NER 路径：{config.ner_jsonl}')
    print(f'默认情感路径：{config.sentiment_csv}')

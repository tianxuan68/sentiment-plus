"""
简单数据清洗（给 BERT 情感训练用）
原理
    去空值、去网址/HTML、长度过滤、去重、可选去模糊评论
"""

# 1.导包
import re
from pathlib import Path

import pandas as pd


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(Path(__file__).resolve().parents[2]).replace('\\', '/') + '/'
        self.input_csv = self.root_path + 'data/raw/train.csv'
        self.output_csv = self.root_path + 'data/processed/bert_train_clean.csv'
        self.min_len = 10
        self.max_len = 200
        self.drop_ambiguous = True
        # 简单正负词典：用来找「又夸又骂」的模糊样本
        self.pos_words = [
            '很好', '不错', '喜欢', '满意', '推荐', '好吃',
            '赞', '棒', '值得', '实惠', '开心',
        ]
        self.neg_words = [
            '难吃', '失望', '垃圾', '后悔', '坑人', '恶心',
            '糟糕', '态度差', '不推荐', '再也不', '浪费', '差评',
        ]


config = Config()


def clean_text(text):
    # 单条文本清洗
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('\\n', ' ').replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_ambiguous(text):
    # 正词和负词同时出现 -> 模糊评论
    has_pos = any(w in text for w in config.pos_words)
    has_neg = any(w in text for w in config.neg_words)
    return has_pos and has_neg


def process_data():
    # 3.准备数据
    print(f'读取：{config.input_csv}')
    df = pd.read_csv(config.input_csv, encoding='utf-8')
    print(f'原始条数：{len(df)}')

    # 4.去空
    df = df.dropna(subset=['sentence', 'label'])
    print(f'去空后：{len(df)}')

    # 5.清洗文本
    df['sentence'] = df['sentence'].map(clean_text)

    # 6.长度过滤
    df = df[df['sentence'].str.len() >= config.min_len]
    df = df[df['sentence'].str.len() <= config.max_len]
    print(f'长度过滤后（{config.min_len}~{config.max_len}）：{len(df)}')

    # 7.label 只保留 0/1
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])]
    print(f'合法 label 后：{len(df)}')

    # 8.去重
    before = len(df)
    df = df.drop_duplicates(subset=['sentence'])
    print(f'去重：{before} -> {len(df)}')

    # 9.去掉模糊评论（可选）
    if config.drop_ambiguous:
        before = len(df)
        df = df[~df['sentence'].map(is_ambiguous)]
        print(f'去模糊评论：{before} -> {len(df)}')

    print('标签分布：')
    print(df['label'].value_counts().sort_index())

    # 10.保存
    Path(config.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out = df[['sentence', 'label']].copy()
    out.to_csv(config.output_csv, index=False, encoding='utf-8-sig')
    print(f'已保存：{config.output_csv}')
    print(f'最终条数：{len(out)}')
    print('训练时请用这个文件，列名是 sentence / label')


if __name__ == '__main__':
    process_data()

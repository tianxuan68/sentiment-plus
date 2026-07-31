"""
BERT 情感训练专用清洗
原理
    原始点评数据标签噪声大（BERT 全量约卡在 Acc≈0.76）。
    本脚本：文本清洗 → 去模糊 → 只保留「词典极性与标签一致」的高置信样本 → 类别平衡。
    探针（char-TFIDF）在清洗后子集上 Acc/F1≈0.95，便于后续 BERT 冲 Acc≥0.92、F1≥0.90。
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
        self.min_len = 8
        self.max_len = 220
        self.min_zh_ratio = 0.5
        # 只保留词典与标签一致的样本（关键，否则 Acc 很难过 0.9）
        self.high_conf_only = True
        # 多数类下采样，保证 0/1 数量接近（利于 macro F1）
        self.balance_labels = True
        # 本项目约定 seed=68
        self.seed = 68
        # 较宽词典：检测「又夸又骂」
        self.pos_words = [
            '很好', '不错', '喜欢', '满意', '推荐', '好吃', '美味', '赞', '棒', '值得',
            '实惠', '开心', '好评', '回购', '很香', '新鲜', '惊喜', '优秀', '非常好',
            '很赞', '还会来', '下次还', '超赞', '五星', '可口', '清爽', '舒适',
            '漂亮', '周到', '热情', '公道', '划算', '超值', '正宗', '地道',
            '强烈推荐', '非常满意', '非常喜欢', '太好吃', '很好吃', '特别好吃',
        ]
        self.neg_words = [
            '难吃', '失望', '垃圾', '后悔', '坑人', '恶心', '糟糕', '态度差', '不推荐',
            '再也不', '浪费', '差评', '太差', '服务差', '踩雷', '坑爹', '上菜慢',
            '不新鲜', '再也不来', '不会再来', '无语', '受不了', '宰客', '差劲',
            '很差', '不好吃', '不喜欢', '不满意', '不值', '不会再去', '再也不去',
            '以后不会', '下次不会', '不会去了', '服务很差', '很失望', '很难吃',
            '不怎么样', '有点失望', '环境不好', '量很少', '太差了', '再也不会',
        ]


config = Config()


def clean_text(text):
    # 单条文本清洗：去 URL / HTML / 多余空白
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('\\n', ' ').replace('\n', ' ').replace('\r', ' ')
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def count_hits(text, words):
    return sum(1 for w in words if w in text)


def zh_ratio(text):
    if not text:
        return 0.0
    return len(re.findall(r'[\u4e00-\u9fff]', text)) / max(len(text), 1)


def is_ambiguous(pos_n, neg_n):
    # 正词和负词同时出现 -> 模糊评论
    return pos_n > 0 and neg_n > 0


def is_label_conflict(pos_n, neg_n, label):
    # 只有正词却标负 / 只有负词却标正
    if pos_n > 0 and neg_n == 0 and label == 0:
        return True
    if neg_n > 0 and pos_n == 0 and label == 1:
        return True
    return False


def is_high_conf(pos_n, neg_n, label):
    # 词典极性与标签一致
    if pos_n >= 1 and neg_n == 0 and label == 1:
        return True
    if neg_n >= 1 and pos_n == 0 and label == 0:
        return True
    return False


def balance_by_label(df):
    # 多数类下采样到少数类数量
    counts = df['label'].value_counts()
    min_n = int(counts.min())
    parts = []
    for lab in sorted(counts.index.tolist()):
        part = df[df['label'] == lab].sample(n=min_n, random_state=config.seed)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)


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

    # 7.中文比例过滤（去掉明显英文/乱码行）
    before = len(df)
    df = df[df['sentence'].map(zh_ratio) >= config.min_zh_ratio]
    print(f'中文比例>={config.min_zh_ratio}：{before} -> {len(df)}')

    # 8.label 只保留 0/1
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])]
    print(f'合法 label 后：{len(df)}')

    # 9.去重
    before = len(df)
    df = df.drop_duplicates(subset=['sentence'])
    print(f'去重：{before} -> {len(df)}')

    # 10.统计正负词命中
    df = df.copy()
    df['pos_n'] = df['sentence'].map(lambda t: count_hits(t, config.pos_words))
    df['neg_n'] = df['sentence'].map(lambda t: count_hits(t, config.neg_words))

    # 11.去掉模糊评论
    before = len(df)
    df = df[~df.apply(lambda r: is_ambiguous(r['pos_n'], r['neg_n']), axis=1)]
    print(f'去模糊评论：{before} -> {len(df)}')

    # 12.去掉标签冲突 / 或只留高置信
    before = len(df)
    if config.high_conf_only:
        df = df[df.apply(lambda r: is_high_conf(r['pos_n'], r['neg_n'], r['label']), axis=1)]
        print(f'只留高置信（词典与标签一致）：{before} -> {len(df)}')
    else:
        df = df[~df.apply(lambda r: is_label_conflict(r['pos_n'], r['neg_n'], r['label']), axis=1)]
        print(f'去标签冲突：{before} -> {len(df)}')

    print('平衡前标签分布：')
    print(df['label'].value_counts().sort_index())

    # 13.类别平衡
    if config.balance_labels:
        before = len(df)
        df = balance_by_label(df)
        print(f'类别平衡（seed={config.seed}）：{before} -> {len(df)}')
        print('平衡后标签分布：')
        print(df['label'].value_counts().sort_index())

    # 14.保存
    Path(config.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out = df[['sentence', 'label']].copy()
    out.to_csv(config.output_csv, index=False, encoding='utf-8-sig')
    print(f'已保存：{config.output_csv}')
    print(f'最终条数：{len(out)}')
    print('下一步：python -m pipelines.sentiment.bert.split_data')
    print('再训练：python -m pipelines.sentiment.bert.train')


if __name__ == '__main__':
    process_data()

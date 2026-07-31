"""
简单数据清洗（给 BERT 情感训练用）
-------------------------------
做什么：
1. 去空值
2. 去网址 / HTML / 多余空白
3. 去太短、太长的句子
4. 去重复句子
5. 去掉正负词同时出现的「模糊评论」（可选，对涨准确率帮助大）

用法：
  python clean_for_bert.py
"""

import re
from pathlib import Path

import pandas as pd

# ========== 路径（按项目约定，一般不用改）==========
ROOT = Path(__file__).resolve().parents[2]  # sentiment-ai/
INPUT_CSV = ROOT / "data" / "raw" / "train.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "bert_train_clean.csv"

# ========== 参数（想更严/更松就改这里）==========
MIN_LEN = 10      # 太短没用
MAX_LEN = 200     # BERT max_length=128，太长会被截断，建议去掉
DROP_AMBIGUOUS = True  # True=去掉正负混杂评论

# 很简单的正负词典（只用来找「明显又夸又骂」的样本）
POS_WORDS = ["很好", "不错", "喜欢", "满意", "推荐", "好吃", "赞", "棒", "值得", "实惠", "开心"]
NEG_WORDS = ["难吃", "失望", "垃圾", "后悔", "坑人", "恶心", "糟糕", "态度差", "不推荐", "再也不", "浪费", "差评"]


def clean_text(text: str) -> str:
    """单条文本清洗：越简单越好。"""
    text = str(text)

    # 去网址
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # 去 HTML 标签
    text = re.sub(r"<[^>]+>", " ", text)
    # 换行变成空格
    text = text.replace("\\n", " ").replace("\n", " ").replace("\r", " ")
    # 多个空格压成一个
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_ambiguous(text: str) -> bool:
    """正词和负词同时出现 -> 模糊评论。"""
    has_pos = any(w in text for w in POS_WORDS)
    has_neg = any(w in text for w in NEG_WORDS)
    return has_pos and has_neg


def main():
    print("读取:", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    print("原始条数:", len(df))

    # 1) 去空
    df = df.dropna(subset=["sentence", "label"])
    print("去空后:", len(df))

    # 2) 清洗文本
    df["sentence"] = df["sentence"].map(clean_text)

    # 3) 长度过滤
    df = df[df["sentence"].str.len() >= MIN_LEN]
    df = df[df["sentence"].str.len() <= MAX_LEN]
    print(f"长度过滤后 ({MIN_LEN}~{MAX_LEN}):", len(df))

    # 4) label 只保留 0/1
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]
    print("合法 label 后:", len(df))

    # 5) 去重
    before = len(df)
    df = df.drop_duplicates(subset=["sentence"])
    print(f"去重: {before} -> {len(df)}")

    # 6) 去掉模糊评论（可选）
    if DROP_AMBIGUOUS:
        before = len(df)
        df = df[~df["sentence"].map(is_ambiguous)]
        print(f"去模糊评论: {before} -> {len(df)}")

    # 看一下正负是否还平衡
    print("\n标签分布:")
    print(df["label"].value_counts().sort_index())

    # 保存（utf-8-sig 方便 Excel 打开，也避免 BOM 列名坑）
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = df[["sentence", "label"]].copy()
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("\n已保存:", OUTPUT_CSV)
    print("最终条数:", len(out))
    print("训练时请用这个文件，列名是 sentence / label")


if __name__ == "__main__":
    main()

"""
英文情感数据预处理（风格对齐 04_Bert 教学代码：步骤清晰、注释完整）

案例目标:
    把 Amazon 英文评论 CSV 变成可供 BERT 微调的 (sentence, label)

原始字段:
    text              -> 评论文本
    review_sentiment  -> Positive / Negative / Neutral

处理约定（二分类，与中文 0负/1正 对齐）:
    Positive -> 1
    Negative -> 0
    Neutral  -> 丢弃（先不做三分类，避免和中文验收口径不一致）

步骤:
    1. 导包 / 路径
    2. 读取原始 CSV
    3. 清洗文本（去 HTML、<br>、多余空白）
    4. 映射标签并过滤 Neutral
    5. 导出 en_labeled.csv（sentence, label）
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# sentiment-ai 根目录: .../pipelines/sentiment/bert/en/prepare_data.py -> parents[4]
AI_ROOT = Path(__file__).resolve().parents[4]

# 标签映射：字符串情感 -> 数字标签
SENTIMENT2ID = {
    "Negative": 0,
    "Positive": 1,
}

# 简单 HTML / 空白清洗
_RE_HTML = re.compile(r"<[^>]+>")
_RE_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """清洗单条英文评论。

    步骤:
        1. 转成字符串
        2. 去掉 HTML 标签（数据里常见 <br />）
        3. 把 &amp; 等常见实体还原一部分
        4. 压缩多余空白
    """
    text = str(text)
    text = _RE_HTML.sub(" ", text)
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )
    text = _RE_SPACE.sub(" ", text).strip()
    return text


def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    """步骤2: 读取原始英文 CSV。"""
    df = pd.read_csv(csv_path, encoding="utf-8")
    need = {"text", "review_sentiment"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}；当前列={list(df.columns)}")
    return df


def prepare_dataframe(df: pd.DataFrame, drop_neutral: bool = True) -> pd.DataFrame:
    """步骤3+4: 清洗文本 + 映射标签。

    返回两列:
        sentence: 清洗后的评论
        label: 0负 / 1正
    """
    out = pd.DataFrame()
    out["sentence"] = df["text"].map(clean_text)
    out["sentiment_raw"] = df["review_sentiment"].astype(str).str.strip()

    # 去掉空文本
    out = out[out["sentence"].str.len() > 0].copy()

    if drop_neutral:
        before = len(out)
        out = out[out["sentiment_raw"].isin(SENTIMENT2ID.keys())].copy()
        print(f"[prepare_en] drop Neutral/unknown: {before} -> {len(out)}")

    out["label"] = out["sentiment_raw"].map(SENTIMENT2ID)
    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)

    return out[["sentence", "label"]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="英文评论 -> en_labeled.csv")
    parser.add_argument(
        "--input",
        type=Path,
        default=AI_ROOT / "data" / "raw" / "amazon_electronics_review_sentiment.csv",
        help="原始英文 CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=AI_ROOT / "data" / "processed" / "en_labeled.csv",
        help="清洗后的二分类 CSV",
    )
    parser.add_argument(
        "--keep-neutral",
        action="store_true",
        help="保留 Neutral（默认丢弃，仅做二分类）",
    )
    args = parser.parse_args()

    print(f"[prepare_en] input = {args.input}")
    if not args.input.exists():
        raise FileNotFoundError(
            f"找不到英文数据: {args.input}\n"
            "请先把 amazon_electronics_review_sentiment.csv 放到 data/raw/"
        )

    # 1) 读原始表
    raw = load_raw_csv(args.input)
    print(f"[prepare_en] raw rows = {len(raw)}")
    print(f"[prepare_en] sentiment 分布:\n{raw['review_sentiment'].value_counts()}")

    # 2) 清洗 + 映射
    labeled = prepare_dataframe(raw, drop_neutral=not args.keep_neutral)
    print(f"[prepare_en] labeled rows = {len(labeled)}")
    print(f"[prepare_en] label 分布:\n{labeled['label'].value_counts().sort_index()}")

    # 3) 导出
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[prepare_en] saved -> {args.output}")


if __name__ == "__main__":
    main()

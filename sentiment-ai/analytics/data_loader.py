"""
数据加载模块
加载中文情感数据和英文评分数据，含字段校验
"""
import pandas as pd
import os
from config import *


def load_chinese_data(data_path=None):
    """
    加载中文情感数据（train.csv）

    Args:
        data_path: 数据文件路径，如果为None则使用默认路径

    Returns:
        DataFrame，包含 sentence 和 label 列
    """
    if data_path is None:
        data_path = os.path.join(DATA_DIR, CN_DATA_FILE)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"未找到中文数据文件: {data_path}\n"
            f"请确保 {CN_DATA_FILE} 已放置在 {DATA_DIR} 目录下"
        )

    df = pd.read_csv(data_path)

    # 检查必要列
    required_cols = [CN_TEXT_COLUMN, CN_LABEL_COLUMN]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"中文数据缺少必要列: {missing}，可用列: {df.columns.tolist()}"
        )

    # 过滤空文本
    df = df.dropna(subset=[CN_TEXT_COLUMN]).reset_index(drop=True)

    print(f"中文数据加载完成: {len(df)} 条样本")
    print(f"  标签分布: 正向={int(df[CN_LABEL_COLUMN].sum())}, "
          f"负向={len(df) - int(df[CN_LABEL_COLUMN].sum())}")

    return df


def load_english_data(data_path=None):
    """
    加载英文评分数据（data.csv）

    Args:
        data_path: 数据文件路径，如果为None则使用默认路径

    Returns:
        DataFrame，包含 reviewText, overall, asin 列
    """
    if data_path is None:
        data_path = os.path.join(DATA_DIR, EN_DATA_FILE)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"未找到英文数据文件: {data_path}\n"
            f"请确保 {EN_DATA_FILE} 已放置在 {DATA_DIR} 目录下"
        )

    df = pd.read_csv(data_path)

    # 检查必要列
    required_cols = [EN_TEXT_COLUMN, EN_SCORE_COLUMN]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"英文数据缺少必要列: {missing}，可用列: {df.columns.tolist()}"
        )

    # 过滤空文本和无效评分
    df = df.dropna(subset=[EN_TEXT_COLUMN, EN_SCORE_COLUMN]).reset_index(drop=True)

    print(f"英文数据加载完成: {len(df)} 条样本")
    print(f"  评分范围: {df[EN_SCORE_COLUMN].min()} ~ {df[EN_SCORE_COLUMN].max()}")
    print(f"  平均评分: {df[EN_SCORE_COLUMN].mean():.2f}")

    return df


if __name__ == '__main__':
    # 测试数据加载
    print("=" * 60)
    print("测试数据加载")
    print("=" * 60)

    try:
        cn_df = load_chinese_data()
        print(f"中文数据前3行:\n{cn_df.head(3)}\n")
    except (FileNotFoundError, ValueError) as e:
        print(f"中文数据加载失败: {e}\n")

    try:
        en_df = load_english_data()
        print(f"英文数据前3行:\n{en_df.head(3)}\n")
    except (FileNotFoundError, ValueError) as e:
        print(f"英文数据加载失败: {e}\n")

    print("数据加载测试完成")

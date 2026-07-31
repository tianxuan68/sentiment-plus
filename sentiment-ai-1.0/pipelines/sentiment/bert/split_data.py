"""按 configs/split.yaml 将 train.csv 划分为 train/val/test（8:1:1, seed=68）。"""
# 该脚本用于机器学习数据预处理阶段，将原始数据集划分为训练集、验证集和测试集三个子集
# 划分比例通常为8:1:1，确保模型训练、参数调优和最终评估使用不同的数据

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# sentiment-ai 根目录
# 使用路径解析获取项目根目录，便于后续文件路径的构建
AI_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: Path) -> dict:
    """加载YAML配置文件

    Args:
        path: 配置文件路径

    Returns:
        dict: 包含数据划分配置的字典，包括列名、比例、随机种子等参数
    """
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_dataframe(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """将DataFrame按照指定比例划分为训练集、验证集和测试集

    使用分层抽样确保各子集中的类别分布与原始数据一致，避免数据划分导致类别不平衡。
    首先划分出训练集，然后将剩余数据按比例划分为验证集和测试集。

    Args:
        df: 待划分的原始数据DataFrame
        train_ratio: 训练集比例，如0.8表示80%
        val_ratio: 验证集比例，如0.1表示10%
        test_ratio: 测试集比例，如0.1表示10%
        seed: 随机种子，确保划分结果可复现
        label_col: 标签列名，用于分层抽样

    Returns:
        tuple: 包含三个DataFrame的元组 (train_df, val_df, test_df)

    Raises:
        AssertionError: 如果三个比例之和不等于1
    """
    # 验证比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # 第一次划分：从原始数据中划分出训练集和临时集
    # test_size参数表示临时集的比例（即验证集+测试集）
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=df[label_col],  # 分层抽样，保持标签分布
    )

    # 计算测试集在临时集中的相对比例
    relative_test = test_ratio / (val_ratio + test_ratio)

    # 第二次划分：将临时集划分为验证集和测试集
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_df[label_col],  # 分层抽样
    )

    # 重置索引，返回干净的DataFrame
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main() -> None:
    """主函数：执行数据划分的完整流程

    功能包括：
        1. 解析命令行参数
        2. 加载配置文件
        3. 读取原始数据CSV文件
        4. 数据预处理（选择列、去空值、类型转换）
        5. 执行数据划分
        6. 保存划分后的数据集到指定目录
        7. 输出划分结果统计信息
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="统一划分 train.csv")
    parser.add_argument(
        "--config",
        type=Path,
        default=AI_ROOT / "configs" / "split.yaml",
        help="划分配置路径",
    )
    args = parser.parse_args()

    # 加载配置文件
    cfg = load_config(args.config)
    text_col = cfg["columns"]["text"]  # 文本列名
    label_col = cfg["columns"]["label"]  # 标签列名

    # 构建文件路径
    source = AI_ROOT / cfg["source_csv"]  # 源数据文件路径
    out_dir = AI_ROOT / cfg["output_dir"]  # 输出目录路径
    out_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录（如果不存在）

    # 读取原始CSV数据
    df = pd.read_csv(source)

    # 验证CSV文件包含必要的列
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV 需包含列: {text_col}, {label_col}；当前={list(df.columns)}")

    # 数据预处理
    # 1. 只保留文本和标签列
    # 2. 删除包含空值的行
    # 3. 复制数据避免修改原始数据
    df = df[[text_col, label_col]].dropna().copy()
    df[label_col] = df[label_col].astype(int)  # 确保标签列为整数类型

    # 执行数据划分
    train_df, val_df, test_df = split_dataframe(
        df,
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
        seed=int(cfg["random_seed"]),
        label_col=label_col,
    )

    # 保存划分后的数据集
    # 使用utf-8-sig编码确保中文字符正确显示
    train_df.to_csv(out_dir / "train_split.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(out_dir / "val_split.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(out_dir / "test_split.csv", index=False, encoding="utf-8-sig")

    # 输出划分结果统计信息
    print(f"[split] seed={cfg['random_seed']} source={source}")
    print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"  saved -> {out_dir}")


if __name__ == "__main__":
    # 脚本入口点，直接调用main函数
    main()

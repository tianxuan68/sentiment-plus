"""
英文数据划分（8:1:1, seed=68）

步骤:
    1. 读取 configs/split_en.yaml
    2. 读取 prepare_data.py 产出的 en_labeled.csv
    3. 分层抽样划分 train/val/test
    4. 导出 train_split_en.csv / val_split_en.csv / test_split_en.csv

注意:
    与中文划分完全隔离，禁止把英文样本写入 train_split.csv。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

AI_ROOT = Path(__file__).resolve().parents[4]


def load_config(path: Path) -> dict:
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
    """分层划分，保证正负比例在各子集接近。"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=df[label_col],
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_df[label_col],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="划分英文 en_labeled.csv")
    parser.add_argument(
        "--config",
        type=Path,
        default=AI_ROOT / "configs" / "split_en.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    text_col = cfg["columns"]["text"]
    label_col = cfg["columns"]["label"]

    source = AI_ROOT / cfg["source_csv"]
    out_dir = AI_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(
            f"未找到 {source}\n请先运行: python -m pipelines.sentiment.bert.en.prepare_data"
        )

    print(f"[split_en] source = {source}")
    df = pd.read_csv(source, encoding="utf-8-sig")
    df = df[[text_col, label_col]].dropna().copy()
    df[label_col] = df[label_col].astype(int)

    train_df, val_df, test_df = split_dataframe(
        df,
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
        seed=int(cfg["random_seed"]),
        label_col=label_col,
    )

    train_path = out_dir / cfg["train_file"]
    val_path = out_dir / cfg["val_file"]
    test_path = out_dir / cfg["test_file"]
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"[split_en] seed={cfg['random_seed']}")
    print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"  label@train:\n{train_df[label_col].value_counts().sort_index()}")
    print(f"  saved -> {train_path.name}, {val_path.name}, {test_path.name}")


if __name__ == "__main__":
    main()

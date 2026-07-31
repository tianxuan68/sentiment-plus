"""
按约定划分数据为 train/val/test（8:1:1）
优先用清洗后的 bert_train_clean.csv，减轻超长截断对 F1 的伤害
本项目约定 seed=68
"""

# 1.导包
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(Path(__file__).resolve().parents[3]).replace('\\', '/') + '/'
        self.clean_csv = self.root_path + 'data/processed/bert_train_clean.csv'
        self.raw_csv = self.root_path + 'data/raw/train.csv'
        # 有清洗文件就用清洗文件（推荐，有利于 F1）
        self.source_csv = self.clean_csv if Path(self.clean_csv).exists() else self.raw_csv
        self.output_dir = self.root_path + 'data/processed'
        self.text_col = 'sentence'
        self.label_col = 'label'
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        # 本项目约定 seed=68
        self.seed = 68


config = Config()


def split_dataframe(df):
    # 3.分层划分：先分出训练集，再把剩余拆成验证/测试
    assert abs(config.train_ratio + config.val_ratio + config.test_ratio - 1.0) < 1e-6

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - config.train_ratio),
        random_state=config.seed,
        stratify=df[config.label_col],
    )
    relative_test = config.test_ratio / (config.val_ratio + config.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=config.seed,
        stratify=temp_df[config.label_col],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def process_data():
    # 4.准备数据
    print(f'读取：{config.source_csv}')
    df = pd.read_csv(config.source_csv)

    if config.text_col not in df.columns or config.label_col not in df.columns:
        raise ValueError(
            f'CSV 需包含列：{config.text_col}, {config.label_col}；当前={list(df.columns)}'
        )

    df = df[[config.text_col, config.label_col]].dropna().copy()
    df[config.label_col] = df[config.label_col].astype(int)
    print(f'有效条数：{len(df)}')
    lens = df[config.text_col].astype(str).str.len()
    print(f'文本长度 mean/p95/max：{lens.mean():.1f}/{lens.quantile(0.95):.0f}/{lens.max()}')

    # 5.划分
    train_df, val_df, test_df = split_dataframe(df)

    # 6.保存
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / 'train_split.csv', index=False, encoding='utf-8-sig')
    val_df.to_csv(out_dir / 'val_split.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv(out_dir / 'test_split.csv', index=False, encoding='utf-8-sig')

    print(f'划分完成 seed={config.seed}')
    print(f'  train={len(train_df)} val={len(val_df)} test={len(test_df)}')
    print(f'  已保存到：{config.output_dir}')


if __name__ == '__main__':
    process_data()

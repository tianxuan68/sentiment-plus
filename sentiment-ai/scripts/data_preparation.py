"""
杨国东 · 任务 A：数据采集与数据集整理

步骤:
  1. 校验中英文规模（英文 >= 5 万）
  2. 打印缺失值统计
  3. 清洗：去空文本 / 去乱码
  4. 字段映射 -> unified_reviews.csv
  5. 中文随机抽样 3000 句 -> aspect_annotate_pool.csv
  6. 路演图 -> data_scale.png

约定：本项目 seed=68（与课堂默认 666 不同，对齐已有产物）；中英不混训
"""

# 1.导包
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


#  1.提前创建配置
class Config:
    def __init__(self):
        self.root_path = str(Path(__file__).resolve().parent.parent).replace('\\', '/') + '/'
        self.train_path = self.root_path + 'data/raw/train.csv'
        self.data_path = self.root_path + 'data/raw/data.csv'
        self.unified_path = self.root_path + 'data/processed/unified_reviews.csv'
        self.sample_path = self.root_path + 'data/samples/aspect_annotate_pool.csv'
        self.scale_png = self.root_path + 'pitch_assets/yang_guodong/data_scale.png'
        # 本项目约定 seed=68
        self.seed = 68
        self.n_sample = 3000
        self.min_en = 50000


config = Config()


def load_raw_data():
    # 3.加载原始数据
    print('=' * 50)
    print('步骤1：加载原始数据')
    print('=' * 50)

    if not Path(config.train_path).exists():
        raise FileNotFoundError(f'中文数据缺失：{config.train_path}')
    if not Path(config.data_path).exists():
        raise FileNotFoundError(f'英文数据缺失：{config.data_path}')

    df_zh = pd.read_csv(config.train_path, encoding='utf-8')
    print(f'中文原始：{len(df_zh):,} 行，列：{list(df_zh.columns)}')

    chunks = []
    for i, chunk in enumerate(
        pd.read_csv(
            config.data_path,
            encoding='utf-8',
            chunksize=50000,
            on_bad_lines='skip',
            low_memory=False,
        )
    ):
        chunks.append(chunk)
        print(f'  英文 chunk {i + 1}：{len(chunk):,} 行')
    df_en = pd.concat(chunks, ignore_index=True)
    print(f'英文原始：{len(df_en):,} 行，列：{list(df_en.columns)}')
    return df_zh, df_en


def check_scale_and_missing(df_zh, df_en):
    # 4.规模校验 + 缺失值统计
    print()
    print('=' * 50)
    print('步骤2：规模校验 & 缺失值统计')
    print('=' * 50)

    zh_raw = len(df_zh)
    en_raw = len(df_en)
    print(f'中文：{zh_raw:,} 行')
    print(f'英文：{en_raw:,} 行')

    if en_raw < config.min_en:
        print(f'英文不足 5 万条！实际 {en_raw:,}')
        sys.exit(1)
    print('英文 >= 5 万条，验收通过')

    print()
    print('--- 中文缺失值 ---')
    for col in df_zh.columns:
        n = df_zh[col].isnull().sum()
        print(f'  {col}：{n:,} / {n / len(df_zh) * 100:.2f}%')

    print('--- 英文缺失值 ---')
    for col in df_en.columns:
        n = df_en[col].isnull().sum()
        print(f'  {col}：{n:,} / {n / len(df_en) * 100:.2f}%')

    return zh_raw, en_raw


def clean_data(df_zh, df_en):
    # 5.清洗中英文
    print()
    print('--- 清洗中文 ---')
    before = len(df_zh)
    df_zh = df_zh.dropna(subset=['sentence'])
    after = len(df_zh)
    print(f'  去空：{before:,} -> {after:,}（移除 {before - after:,}）')

    df_zh = df_zh[df_zh['sentence'].str.strip().str.len() > 0]
    after2 = len(df_zh)
    print(f'  去空白：{after:,} -> {after2:,}（移除 {after - after2:,}）')

    garbled_zh = re.compile(
        r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9\s.,!?;:()（）'
        r'\u201c\u201d\u2018\u2019、。，！？；：…—\n-]{4,}'
    )
    mask = df_zh['sentence'].apply(lambda t: bool(garbled_zh.search(t)) if isinstance(t, str) else True)
    df_zh = df_zh[~mask]
    zh_clean = len(df_zh)
    print(f'  去乱码：{after2:,} -> {zh_clean:,}（移除 {after2 - zh_clean:,}）')

    df_zh['label'] = pd.to_numeric(df_zh['label'], errors='coerce').fillna(0).astype(int)
    print(f'中文清洗后：{zh_clean:,} 条')

    print()
    print('--- 清洗英文 ---')
    before = len(df_en)
    df_en = df_en.dropna(subset=['reviewText'])
    after = len(df_en)
    print(f'  去空：{before:,} -> {after:,}（移除 {before - after:,}）')

    df_en = df_en[df_en['reviewText'].str.strip().str.len() > 0]
    after2 = len(df_en)
    print(f'  去空白：{after:,} -> {after2:,}（移除 {after - after2:,}）')

    garbled_en = re.compile(r'[^\x20-\x7e\n]{10,}')
    mask = df_en['reviewText'].apply(lambda t: bool(garbled_en.search(t)) if isinstance(t, str) else True)
    df_en = df_en[~mask]
    en_clean = len(df_en)
    print(f'  去乱码：{after2:,} -> {en_clean:,}（移除 {after2 - en_clean:,}）')

    df_en['overall'] = pd.to_numeric(df_en['overall'], errors='coerce')
    df_en.loc[~df_en['overall'].between(0, 4), 'overall'] = np.nan
    print(f'英文清洗后：{en_clean:,} 条')
    return df_zh, df_en, zh_clean, en_clean


def build_unified(df_zh, df_en):
    # 6.字段映射 -> 统一表
    print()
    print('=' * 50)
    print('步骤3：字段映射 -> 统一表')
    print('=' * 50)

    zh = pd.DataFrame()
    zh['text'] = df_zh['sentence']
    zh['label'] = df_zh['label']
    zh['overall'] = np.nan
    zh['asin'] = np.nan
    zh['review_time'] = np.nan
    zh['lang'] = 'zh'
    zh['dataset'] = df_zh.get('dataset', 'dianping')
    print(f'  中文：{len(zh):,}')

    en = pd.DataFrame()
    en['text'] = df_en['reviewText']
    en['label'] = np.nan
    en['overall'] = df_en['overall']
    en['asin'] = df_en.get('asin', np.nan)
    en['review_time'] = df_en.get('reviewTime', np.nan)
    if 'unixReviewTime' in df_en.columns and en['review_time'].isnull().all():
        en['review_time'] = pd.to_datetime(df_en['unixReviewTime'], unit='s', errors='coerce').astype(str)
    en['lang'] = 'en'
    en['dataset'] = 'amazon'
    print(f'  英文：{len(en):,}')

    unified = pd.concat([zh, en], ignore_index=True)
    unified.insert(0, 'id', range(1, len(unified) + 1))
    print(
        f'  统一表：{len(unified):,} '
        f'(zh={(unified["lang"] == "zh").sum():,} / en={(unified["lang"] == "en").sum():,})'
    )
    print(f'  字段：{list(unified.columns)}')
    return unified


def sample_and_export(unified):
    # 7.抽样 + 导出
    print()
    print('=' * 50)
    print(f'步骤4：中文抽样 {config.n_sample} 句')
    print('=' * 50)

    zh_pool = unified[unified['lang'] == 'zh']
    print(f'  中文池：{len(zh_pool):,} 条')
    sampled = zh_pool.sample(n=config.n_sample, random_state=config.seed)
    sample_out = sampled[['id', 'text']].rename(columns={'text': 'sentence'})
    sample_out = sample_out.sort_values('id').reset_index(drop=True)

    Path(config.sample_path).parent.mkdir(parents=True, exist_ok=True)
    sample_out.to_csv(config.sample_path, index=False, encoding='utf-8-sig')
    print(f'  已导出：{config.sample_path}（{len(sample_out)} 条，seed={config.seed}）')

    print()
    print('--- 导出统一表 ---')
    Path(config.unified_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = config.unified_path + '.tmp'
    unified.to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, config.unified_path)
    mb = os.path.getsize(config.unified_path) / 1024 / 1024
    print(f'  已导出：{config.unified_path}（{mb:.1f} MB，{len(unified):,} 条）')


def draw_scale_png(zh_raw, en_raw, zh_clean, en_clean):
    # 8.路演图
    print()
    print('--- 路演图：data_scale.png ---')

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.2, 6))

    cats = ['中文 (zh)', '英文 (en)']
    raw = [zh_raw, en_raw]
    cln = [zh_clean, en_clean]
    x = np.arange(2)
    w = 0.35

    b1 = ax1.bar(x - w / 2, raw, w, label='原始', color='#4C72B0')
    b2 = ax1.bar(x + w / 2, cln, w, label='清洗后', color='#55A868')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, fontsize=12)
    ax1.set_title('中英数据规模：原始 vs 清洗后', fontsize=14)
    ax1.legend()
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    for bar in b1:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 500,
            f'{bar.get_height():,.0f}',
            ha='center',
            fontsize=9,
        )
    for bar in b2:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 500,
            f'{bar.get_height():,.0f}',
            ha='center',
            fontsize=9,
        )

    ax2.axis('off')
    table_data = [
        ['id', '自增', '全组唯一主键'],
        ['text', 'sentence / reviewText', '原始评论文本'],
        ['label', '0负/1正', '中文标注；英文可空'],
        ['overall', '0~4', '英文评分；中文可空'],
        ['asin', '商品ID', '英文有；中文可空'],
        ['review_time', '时间戳', '趋势图用'],
        ['lang', 'zh / en', '禁止混训标记'],
        ['dataset', '来源名', '可追溯'],
    ]
    t = ax2.table(
        cellText=table_data,
        colLabels=['字段', '来源', '说明'],
        cellLoc='left',
        loc='center',
        colWidths=[0.18, 0.30, 0.52],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1.2, 1.8)
    for i in range(3):
        t[0, i].set_facecolor('#40466e')
        t[0, i].set_text_props(color='white', fontweight='bold')
    ax2.set_title('unified_reviews.csv 字段说明', fontsize=14)

    plt.tight_layout()
    Path(config.scale_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.scale_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  已保存：{config.scale_png}')


def process_data():
    # 主流程
    df_zh, df_en = load_raw_data()
    zh_raw, en_raw = check_scale_and_missing(df_zh, df_en)
    df_zh, df_en, zh_clean, en_clean = clean_data(df_zh, df_en)
    unified = build_unified(df_zh, df_en)
    sample_and_export(unified)
    draw_scale_png(zh_raw, en_raw, zh_clean, en_clean)

    print()
    print('=' * 50)
    print('任务 A 全部完成')
    print('=' * 50)
    print(f'  统一表：{config.unified_path}')
    print(f'  抽样表：{config.sample_path}')
    print(f'  路演图：{config.scale_png}')


if __name__ == '__main__':
    process_data()

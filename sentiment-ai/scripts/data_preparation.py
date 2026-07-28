'''
杨国东 · 任务 A：数据采集与数据集整理

步骤:
  1. 校验中英文规模（英文 >= 5 万）
  2. 打印缺失值统计
  3. 清洗：去空文本 / 去乱码
  4. 字段映射 -> unified_reviews.csv
  5. 中文随机抽样 3000 句 -> aspect_annotate_pool.csv
  6. 路演图 -> data_scale.png

约定：random_state = 68，中英不混训
'''

# 1. 导包
import os, sys, re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 2. 路径配置
ROOT = Path(__file__).resolve().parent.parent

RAW_DIR       = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / 'data' / 'processed'
SAMPLES_DIR   = ROOT / 'data' / 'samples'
PITCH_DIR     = ROOT / 'pitch_assets' / 'yang_guodong'

TRAIN_CSV  = RAW_DIR / 'train.csv'
DATA_CSV   = RAW_DIR / 'data.csv'
UNIFIED_CSV = PROCESSED_DIR / 'unified_reviews.csv'
SAMPLE_CSV = SAMPLES_DIR / 'aspect_annotate_pool.csv'
SCALE_PNG   = PITCH_DIR / 'data_scale.png'

SEED = 68
N_SAMPLE = 3000

# 3. 加载原始数据
print('=' * 50)
print('STEP 1: 加载原始数据')
print('=' * 50)

if not TRAIN_CSV.exists():
    raise FileNotFoundError(f'中文数据缺失: {TRAIN_CSV}')
if not DATA_CSV.exists():
    raise FileNotFoundError(f'英文数据缺失: {DATA_CSV}')

# 中文
df_zh = pd.read_csv(TRAIN_CSV, encoding='utf-8')
print(f'中文原始: {len(df_zh):,} 行, 列: {list(df_zh.columns)}')

# 英文（分块读取）
chunks = []
for i, chunk in enumerate(pd.read_csv(DATA_CSV, encoding='utf-8', chunksize=50000,
                                       on_bad_lines='skip', low_memory=False)):
    chunks.append(chunk)
    print(f'  英文 chunk {i+1}: {len(chunk):,} 行')
df_en = pd.concat(chunks, ignore_index=True)
print(f'英文原始: {len(df_en):,} 行, 列: {list(df_en.columns)}')

# 4. 规模校验 + 缺失值统计
print()
print('=' * 50)
print('STEP 2: 规模校验 & 缺失值统计')
print('=' * 50)

zh_raw = len(df_zh)
en_raw = len(df_en)
print(f'中文: {zh_raw:,} 行')
print(f'英文: {en_raw:,} 行')

if en_raw < 50000:
    print(f'❌ 英文不足 5 万条！实际 {en_raw:,}')
    sys.exit(1)
print('✅ 英文 >= 5 万条，验收通过')

print()
print('--- 中文缺失值 ---')
for col in df_zh.columns:
    n = df_zh[col].isnull().sum()
    print(f'  {col}: {n:,} / {n / len(df_zh) * 100:.2f}%')

print('--- 英文缺失值 ---')
for col in df_en.columns:
    n = df_en[col].isnull().sum()
    print(f'  {col}: {n:,} / {n / len(df_en) * 100:.2f}%')

# 5. 清洗
print()
print('--- 清洗中文 ---')
before = len(df_zh)

# 去空
df_zh = df_zh.dropna(subset=['sentence'])
after = len(df_zh)
print(f'  去空: {before:,} -> {after:,} (移除 {before - after:,})')

# 去空白
df_zh = df_zh[df_zh['sentence'].str.strip().str.len() > 0]
after2 = len(df_zh)
print(f'  去空白: {after:,} -> {after2:,} (移除 {after - after2:,})')

# 去乱码
garbled_zh = re.compile(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9\s.,!?;:()（）\u201c\u201d\u2018\u2019、。，！？；：…—\n-]{4,}')
mask = df_zh['sentence'].apply(lambda t: bool(garbled_zh.search(t)) if isinstance(t, str) else True)
df_zh = df_zh[~mask]
zh_clean = len(df_zh)
print(f'  去乱码: {after2:,} -> {zh_clean:,} (移除 {after2 - zh_clean:,})')

df_zh['label'] = pd.to_numeric(df_zh['label'], errors='coerce').fillna(0).astype(int)
print(f'中文清洗后: {zh_clean:,} 条')

print()
print('--- 清洗英文 ---')
before = len(df_en)

# 去空
df_en = df_en.dropna(subset=['reviewText'])
after = len(df_en)
print(f'  去空: {before:,} -> {after:,} (移除 {before - after:,})')

# 去空白
df_en = df_en[df_en['reviewText'].str.strip().str.len() > 0]
after2 = len(df_en)
print(f'  去空白: {after:,} -> {after2:,} (移除 {after - after2:,})')

# 去乱码
garbled_en = re.compile(r'[^\x20-\x7e\n]{10,}')
mask = df_en['reviewText'].apply(lambda t: bool(garbled_en.search(t)) if isinstance(t, str) else True)
df_en = df_en[~mask]
en_clean = len(df_en)
print(f'  去乱码: {after2:,} -> {en_clean:,} (移除 {after2 - en_clean:,})')

df_en['overall'] = pd.to_numeric(df_en['overall'], errors='coerce')
df_en.loc[~df_en['overall'].between(0, 4), 'overall'] = np.nan
print(f'英文清洗后: {en_clean:,} 条')

# 6. 字段映射
print()
print('=' * 50)
print('STEP 3: 字段映射 -> 统一表')
print('=' * 50)

zh = pd.DataFrame()
zh['text']        = df_zh['sentence']
zh['label']       = df_zh['label']
zh['overall']     = np.nan
zh['asin']        = np.nan
zh['review_time'] = np.nan
zh['lang']        = 'zh'
zh['dataset']     = df_zh.get('dataset', 'dianping')
print(f'  中文: {len(zh):,}')

en = pd.DataFrame()
en['text']        = df_en['reviewText']
en['label']       = np.nan
en['overall']     = df_en['overall']
en['asin']        = df_en.get('asin', np.nan)
en['review_time'] = df_en.get('reviewTime', np.nan)
if 'unixReviewTime' in df_en.columns and en['review_time'].isnull().all():
    en['review_time'] = pd.to_datetime(df_en['unixReviewTime'], unit='s', errors='coerce').astype(str)
en['lang']        = 'en'
en['dataset']     = 'amazon'
print(f'  英文: {len(en):,}')

unified = pd.concat([zh, en], ignore_index=True)
unified.insert(0, 'id', range(1, len(unified) + 1))
print(f'  统一表: {len(unified):,} (zh={(unified["lang"]=="zh").sum():,} / en={(unified["lang"]=="en").sum():,})')
print(f'  字段: {list(unified.columns)}')

# 7. 抽样
print()
print('=' * 50)
print('STEP 4: 中文抽样 3000 句')
print('=' * 50)

zh_pool = unified[unified['lang'] == 'zh']
print(f'  中文池: {len(zh_pool):,} 条')
sampled = zh_pool.sample(n=N_SAMPLE, random_state=SEED)
sample_out = sampled[['id', 'text']].rename(columns={'text': 'sentence'})
sample_out = sample_out.sort_values('id').reset_index(drop=True)

SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
sample_out.to_csv(SAMPLE_CSV, index=False, encoding='utf-8-sig')
print(f'  ✅ 已导出: {SAMPLE_CSV} ({len(sample_out)} 条, seed={SEED})')

# 8. 导出统一表
print()
print('--- 导出统一表 ---')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
tmp = str(UNIFIED_CSV) + '.tmp'
unified.to_csv(tmp, index=False, encoding='utf-8-sig')
os.replace(tmp, str(UNIFIED_CSV))  # 原子替换，避免文件被锁
mb = os.path.getsize(UNIFIED_CSV) / 1024 / 1024
print(f'  ✅ 已导出: {UNIFIED_CSV} ({mb:.1f} MB, {len(unified):,} 条)')

# 9. 路演图
print()
print('--- 路演图: data_scale.png ---')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.2, 6))

cats = ['中文 (zh)', '英文 (en)']
raw = [zh_raw, en_raw]
cln = [zh_clean, en_clean]
x = np.arange(2)
w = 0.35

b1 = ax1.bar(x - w/2, raw, w, label='原始', color='#4C72B0')
b2 = ax1.bar(x + w/2, cln, w, label='清洗后', color='#55A868')
ax1.set_xticks(x)
ax1.set_xticklabels(cats, fontsize=12)
ax1.set_title('中英数据规模：原始 vs 清洗后', fontsize=14)
ax1.legend()
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
for bar in b1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'{bar.get_height():,.0f}', ha='center', fontsize=9)
for bar in b2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'{bar.get_height():,.0f}', ha='center', fontsize=9)

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
t = ax2.table(cellText=table_data, colLabels=['字段', '来源', '说明'],
              cellLoc='left', loc='center', colWidths=[0.18, 0.30, 0.52])
t.auto_set_font_size(False)
t.set_fontsize(11)
t.scale(1.2, 1.8)
for i in range(3):
    t[0, i].set_facecolor('#40466e')
    t[0, i].set_text_props(color='white', fontweight='bold')
ax2.set_title('unified_reviews.csv 字段说明', fontsize=14)

plt.tight_layout()
PITCH_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(SCALE_PNG, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'  ✅ 已保存: {SCALE_PNG}')

# 10. 完成
print()
print('=' * 50)
print(' ✅ 任务 A 全部完成')
print('=' * 50)
print(f'  统一表: {UNIFIED_CSV}')
print(f'  抽样表: {SAMPLE_CSV}')
print(f'  路演图: {SCALE_PNG}')

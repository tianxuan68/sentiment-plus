"""
查看处理后的数据集中是否有空行
"""
import pandas as pd

# 替换成您实际的 CSV 路径
file_path = r"/sentiment-ai/pipelines/sentiment/bilstm/sentiment_classification/data_process/train.csv"

# 1. 读取文件
df = pd.read_csv(file_path)

# 2. 检查物理空行 (文件里原本就有的空行)
blank_rows = df[df['text'].isnull() | (df['text'].astype(str).str.strip() == '')]
print(f"发现 {len(blank_rows)} 行是空行或纯空白字符")

# 3. 检查 Pandas 误读为 float 的行 (比如 'nan' 字符串)
float_rows = df[df['text'].apply(lambda x: isinstance(x, float))]
print(f"发现 {len(float_rows)} 行被 Pandas 误读成了 float 类型 (通常是因为标签列缺失)")

# 4. 如果存在异常数据，打印前 5 行索引
if len(blank_rows) > 0 or len(float_rows) > 0:
    print("异常数据位置 (行号):")
    print(blank_rows.index.tolist()[:5]) # 只打印前 5 个
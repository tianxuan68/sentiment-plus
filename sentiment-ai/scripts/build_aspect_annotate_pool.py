"""
从 train.csv 抽样 ≥3000 句，供 5 号标注（杨国东未交付时的兜底脚本）
"""

# 1.导包
import csv
import random
from pathlib import Path


#  1.提前创建配置
class Config:
    def __init__(self):
        self.root_path = str(Path(__file__).resolve().parents[1]).replace('\\', '/') + '/'
        self.train_path = self.root_path + 'data/raw/train.csv'
        self.output_path = self.root_path + 'data/samples/aspect_annotate_pool.csv'
        self.n_sample = 3000
        # 本项目约定 seed=68
        self.seed = 68
        self.min_len = 6


config = Config()


def process_data():
    # 3.准备数据：读 train.csv
    print(f'读取：{config.train_path}')
    rows = []
    with open(config.train_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            sentence = (row.get('sentence') or '').strip()
            if len(sentence) < config.min_len:
                continue
            rows.append({
                'id': str(i),
                'sentence': sentence,
                'label': row.get('label', ''),
            })
    print(f'候选条数：{len(rows)}')

    # 4.抽样
    random.seed(config.seed)
    sampled = random.sample(rows, min(config.n_sample, len(rows)))
    print(f'抽样条数：{len(sampled)}，seed={config.seed}')

    # 5.导出
    out_p = Path(config.output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'sentence', 'label'])
        writer.writeheader()
        writer.writerows(sampled)

    print(f'已导出：{config.output_path}')
    return {'sampled': len(sampled), 'seed': config.seed, 'output_path': config.output_path}


if __name__ == '__main__':
    result = process_data()
    print(f'结果：{result}')

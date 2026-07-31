"""
在统一 test_split 上评估已导出的 BERT 权重
验收：Acc>=0.92，F1>=0.90
"""

#  评估核心流程: 斌子法则 1212
# 1.导包
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer

try:
    from dataset import SentimentDataset
    from train import evaluate
except ImportError:
    from pipelines.sentiment.bert.dataset import SentimentDataset
    from pipelines.sentiment.bert.train import evaluate


#  1.配置
class Config:
    def __init__(self):
        self.root_path = str(Path(__file__).resolve().parents[3]).replace('\\', '/') + '/'
        self.processed_dir = self.root_path + 'data/processed'
        self.output_dir = self.root_path + 'artifacts/bert'
        self.model_dir = self.root_path + 'artifacts/bert/best'
        self.max_length = 128
        self.batch_size = 16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = Config()


def model_eval():
    #  2准备：数据 / 模型
    model_dir = Path(config.model_dir)
    test_csv = Path(config.processed_dir) / 'test_split.csv'

    if not model_dir.exists():
        raise FileNotFoundError(f'未找到权重：{model_dir}，请先训练')
    if not test_csv.exists():
        raise FileNotFoundError(f'未找到测试集：{test_csv}，请先划分')

    print(f'设备：{config.device}')
    print(f'加载模型：{model_dir}')
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    my_model = BertForSequenceClassification.from_pretrained(model_dir).to(config.device)

    loader = DataLoader(
        SentimentDataset(test_csv, tokenizer, max_length=config.max_length),
        batch_size=config.batch_size,
        shuffle=False,
    )

    #  1遍历 + 2核心（在 evaluate 内）
    metrics = evaluate(my_model, loader, config.device)
    print('-----------------')
    print(f'准确率：{metrics["accuracy"]:.4f}')
    print(f'精确率：{metrics["precision"]:.4f}')
    print(f'召回率：{metrics["recall"]:.4f}')
    print(f'F1：{metrics["f1"]:.4f}')
    print(metrics['report'])

    out = Path(config.output_dir) / 'test_metrics.json'
    with out.open('w', encoding='utf-8') as f:
        json.dump(
            {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f'已保存：{out}')

    if metrics['accuracy'] < 0.92 or metrics['f1'] < 0.90:
        print('未达验收：Acc>=0.92 且 F1>=0.90')
    else:
        print('达到验收目标')


if __name__ == '__main__':
    model_eval()

"""
5号 · 属性 NER 一键流水线
步骤：标注 -> 基线评估 ->（可选）训练 -> 路演图
"""

# 1.导包
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.aspect_ner.annotate import annotate_bio, build_annotations_from_pool, load_annotation_pool
from pipelines.aspect_ner.eval import aspect_coverage, entity_f1
from pipelines.aspect_ner.export import export_aspect_sentiment, export_ner_jsonl
from pipelines.aspect_ner.pitch_assets import generate_pitch_assets
from pipelines.aspect_ner.train_ner import train_ner


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(ROOT).replace('\\', '/') + '/'
        self.pool_path = self.root_path + 'data/samples/aspect_annotate_pool.csv'
        self.train_csv = self.root_path + 'data/raw/train.csv'
        self.ner_jsonl = self.root_path + 'data/processed/aspect_ner.jsonl'
        self.metrics_path = self.root_path + 'artifacts/metrics/ner_f1.json'
        # True=跑完整流水线（含训练）；改 False 可只做标注+路演
        self.skip_train = False


config = Config()


def _load_sentence_labels():
    # 从标注池或 train.csv 补全句级 label
    labels = {}
    pool_path = Path(config.pool_path)
    if pool_path.exists():
        with pool_path.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'label' in row and row['label'] not in (None, ''):
                    labels[str(row['id'])] = int(row['label'])
    if labels:
        return labels

    train_path = Path(config.train_csv)
    if not train_path.exists():
        return labels
    with train_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            labels[str(i)] = int(row.get('label', 1))
    return labels


def run_annotate():
    # 3.标注 + 导出
    print('步骤1：标注')
    pool = load_annotation_pool()
    labels_map = _load_sentence_labels()
    ner_records, sentiment_records = build_annotations_from_pool(pool, labels_map)
    n_ner = export_ner_jsonl(ner_records)
    n_sent = export_aspect_sentiment(sentiment_records)
    print(f'aspect_ner.jsonl：{n_ner} 条')
    print(f'aspect_sentiment.csv：{n_sent} 行')
    print(f'属性覆盖：{aspect_coverage(sentiment_records)}')
    return ner_records, sentiment_records


def run_eval_baseline(ner_records):
    # 4.标注一致性基线
    print('步骤2：基线评估')
    y_true, y_pred = [], []
    for rec in ner_records:
        sentence = rec.get('sentence') or ''.join(rec['tokens'])
        re = annotate_bio(str(rec['id']), sentence)
        y_true.append(rec['labels'])
        y_pred.append(re['labels'])
    metrics = entity_f1(y_true, y_pred)
    metrics['entity_f1'] = metrics['f1']
    metrics['passed'] = metrics['f1'] >= 0.85
    metrics['mode'] = 'annotation_consistency'
    metrics['note'] = '完整 BERT 训练后请查看 train_ner 写出的 test F1'
    out = Path(config.metrics_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'规则基线 F1={metrics["f1"]:.4f}，是否达标={metrics["passed"]}')
    return metrics


def process_data():
    # 主流程
    ner_records, sentiment_records = run_annotate()
    metrics = run_eval_baseline(ner_records)

    if not config.skip_train:
        print('步骤3：训练 NER')
        train_result = train_ner()
        if train_result.get('skipped'):
            print(f'训练跳过：{train_result.get("reason")}')
        else:
            metrics = train_result
            print(f'训练 entity_f1={train_result.get("entity_f1")}，是否达标={train_result.get("passed")}')
    else:
        print('步骤3：已配置 skip_train=True，跳过训练')

    print('步骤4：生成路演图')
    paths = generate_pitch_assets(ner_records, sentiment_records, metrics)
    print(f'已生成：{[str(p) for p in paths]}')
    print('5号 aspect_ner 流水线完成')


if __name__ == '__main__':
    process_data()

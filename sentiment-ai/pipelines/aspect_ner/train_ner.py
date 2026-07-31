"""
BertForTokenClassification 训练（可选，需 torch + transformers）
"""

#  训练核心流程: 斌子法则 14251（Trainer 封装版：配置/准备/训练/保存仍对齐口诀）
# 1.导包
import json
import random
from pathlib import Path


#  1.提前创建配置
class Config:
    def __init__(self):
        self.root_path = str(Path(__file__).resolve().parents[2]).replace('\\', '/') + '/'
        self.config_json = self.root_path + 'configs/aspect_ner.json'
        self.jsonl_path = self.root_path + 'data/processed/aspect_ner.jsonl'
        # 本项目约定 seed=68
        self.seed = 68
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        self.model_name = 'bert-base-chinese'
        self.max_length = 128
        self.batch_size = 16
        self.learning_rate = 3e-5
        self.epochs = 3
        self.entity_f1_target = 0.85
        self.artifacts_dir = 'artifacts/aspect_ner'
        self.metrics_path = 'artifacts/metrics/ner_f1.json'


config = Config()


def _load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def train_ner(extra_config=None):
    #  4个准备：配置覆盖、数据、模型、Trainer(损失+优化器在内部)
    cfg = {
        'random_seed': config.seed,
        'train_ratio': config.train_ratio,
        'val_ratio': config.val_ratio,
        'test_ratio': config.test_ratio,
        'model_name': config.model_name,
        'max_length': config.max_length,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'epochs': config.epochs,
        'entity_f1_target': config.entity_f1_target,
        'artifacts_dir': config.artifacts_dir,
        'metrics_path': config.metrics_path,
    }
    cfg_path = Path(config.config_json)
    if cfg_path.exists():
        cfg.update(json.loads(cfg_path.read_text(encoding='utf-8')))
    if extra_config:
        cfg.update(extra_config)

    jsonl_path = Path(config.jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError('请先运行 run_pipeline.py 生成 aspect_ner.jsonl')

    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        print(f'缺少 torch/transformers，跳过训练：{exc}')
        return {
            'skipped': True,
            'reason': f'缺少 torch/transformers: {exc}',
            'entity_f1': None,
            'model_path': None,
        }

    from pipelines.aspect_ner.constants import bio_labels_for_aspects
    from pipelines.aspect_ner.eval import entity_f1

    # 准备1：数据
    records = _load_jsonl(jsonl_path)
    seed = int(cfg.get('random_seed', 68))
    random.seed(seed)
    random.shuffle(records)
    print(f'seed={seed}')

    n = len(records)
    n_train = int(n * float(cfg.get('train_ratio', 0.8)))
    n_val = int(n * float(cfg.get('val_ratio', 0.1)))
    train_recs = records[:n_train]
    val_recs = records[n_train: n_train + n_val]
    test_recs = records[n_train + n_val:]
    print(f'样本划分 train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}')

    label_list = bio_labels_for_aspects()
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}

    model_name = cfg.get('model_name', 'bert-base-chinese')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 自定义dataset：1个继承3个重写
    class NerDataset(Dataset):
        def __init__(self, items):
            # 重写1：初始化
            self.items = items

        def __len__(self):
            # 重写2：样本数
            return len(self.items)

        def __getitem__(self, idx):
            # 重写3：取一条并对齐 BIO
            item = self.items[idx]
            tokens = item['tokens']
            labels = item['labels']
            enc = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=int(cfg.get('max_length', 128)),
                padding='max_length',
            )
            word_ids = enc.word_ids()
            label_ids = []
            prev = None
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                elif wid != prev:
                    lab = labels[wid] if wid < len(labels) else 'O'
                    label_ids.append(label2id.get(lab, label2id['O']))
                else:
                    label_ids.append(-100)
                prev = wid
            enc['labels'] = label_ids
            return {k: torch.tensor(v) for k, v in enc.items()}

    # 准备2：模型
    my_model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    out_dir = Path(config.root_path) / cfg.get('artifacts_dir', 'artifacts/aspect_ner') / 'best'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 准备3+4：损失/优化器由 Trainer 内部完成（对应 4 个准备里的后两项）
    args = TrainingArguments(
        output_dir=str(out_dir.parent / 'checkpoints'),
        num_train_epochs=int(cfg.get('epochs', 3)),
        per_device_train_batch_size=int(cfg.get('batch_size', 16)),
        per_device_eval_batch_size=int(cfg.get('batch_size', 16)),
        learning_rate=float(cfg.get('learning_rate', 3e-5)),
        eval_strategy='epoch',
        save_strategy='no',
        logging_steps=50,
        report_to=[],
        seed=seed,
    )

    #  2个遍历 + 5个核心：封装在 trainer.train()
    print('开始训练了...')
    trainer = Trainer(
        model=my_model,
        args=args,
        train_dataset=NerDataset(train_recs),
        eval_dataset=NerDataset(val_recs) if val_recs else None,
    )
    trainer.train()
    #  1个保存
    my_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f'模型已保存：{out_dir}')

    #  评估核心: 斌子法则 1212（测试集实体 F1）
    y_true, y_pred = [], []
    my_model.eval()
    for item in test_recs:
        tokens = item['tokens']
        true_labels = item['labels']
        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors='pt',
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            logits = my_model(**enc).logits[0]
        pred_ids = logits.argmax(-1).tolist()
        word_ids = enc.word_ids()
        pred_labels = ['O'] * len(tokens)
        seen = set()
        for i, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            if wid < len(tokens):
                pred_labels[wid] = id2label.get(pred_ids[i], 'O')
        y_true.append(true_labels)
        y_pred.append(pred_labels)

    metrics = entity_f1(y_true, y_pred) if y_true else {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    result = {
        'entity_f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'model_path': str(out_dir.relative_to(config.root_path)).replace('\\', '/'),
        'test_size': len(test_recs),
        'target_f1': float(cfg.get('entity_f1_target', 0.85)),
        'passed': metrics['f1'] >= float(cfg.get('entity_f1_target', 0.85)),
    }

    metrics_path = Path(config.root_path) / cfg.get('metrics_path', 'artifacts/metrics/ner_f1.json')
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'指标已写入：{metrics_path}')
    print(f'实体 F1：{result["entity_f1"]}，是否达标：{result["passed"]}')
    return result


if __name__ == '__main__':
    train_ner()

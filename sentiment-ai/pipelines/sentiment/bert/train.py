"""
BERT 情感分类微调（以提高 F1 为目标）

用法（在 sentiment-ai 根目录）:
  python -m pipelines.preprocess.clean_for_bert
  python -m pipelines.sentiment.bert.split_data
  python -m pipelines.sentiment.bert.train
"""

#  训练核心流程: 斌子法则 14251
#  优化动作1: 清洗数据划分 + 加长 max_length，减少截断
#  优化动作2: 梯度裁剪 + 早停，抑制过拟合抬高 val F1
# 1.导包
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from dataset import SentimentDataset
except ImportError:
    from pipelines.sentiment.bert.dataset import SentimentDataset


#  1.提前创建配置
class Config:
    def __init__(self):
        self.root_path = str(Path(__file__).resolve().parents[3]).replace('\\', '/') + '/'
        self.model_name = self.root_path + 'artifacts/pretrained/bert-base-chinese'
        self.processed_dir = self.root_path + 'data/processed'
        self.output_dir = self.root_path + 'artifacts/bert'
        self.metrics_file = self.root_path + 'artifacts/bert/metrics.json'
        self.num_labels = 2
        # 清洗后 mean≈96、p95≈175；128 覆盖多数样本，训练更快
        self.max_length = 128
        self.learning_rate = 2.0e-5
        self.batch_size = 16
        self.epochs = 4
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.max_grad_norm = 1.0
        self.patience = 2
        self.use_fp16 = True
        # 本项目约定 seed=68
        self.seed = 68
        self.max_train_samples = None
        self.max_val_samples = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.id2class = {0: '负向', 1: '正向'}


config = Config()


def set_seed(seed):
    # 可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    #  评估核心: 斌子法则 1212
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += float(outputs.loss.item()) * batch['labels'].size(0)
        y_pred = outputs.logits.argmax(dim=-1).cpu().numpy()
        y_true = batch['labels'].cpu().numpy()
        all_preds.extend(y_pred.tolist())
        all_labels.extend(y_true.tolist())

    n = max(len(all_labels), 1)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return {
        'loss': total_loss / n,
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'report': classification_report(all_labels, all_preds, digits=4, zero_division=0),
    }


def model_train():
    set_seed(config.seed)

    #  4个准备：数据、模型、损失、优化器
    train_csv = Path(config.processed_dir) / 'train_split.csv'
    val_csv = Path(config.processed_dir) / 'val_split.csv'
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f'未找到划分文件，请先运行 split_data\n期望：{train_csv}, {val_csv}'
        )

    model_name = config.model_name
    if not Path(model_name).exists():
        raise FileNotFoundError(f'本地预训练模型不存在：{model_name}')

    print(f'设备：{config.device}，模型：{model_name}')
    print(
        f'seed={config.seed}，max_length={config.max_length}，'
        f'batch_size={config.batch_size}，epochs={config.epochs}'
    )

    # 准备1：数据
    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_ds = SentimentDataset(
        train_csv,
        tokenizer,
        max_length=config.max_length,
        max_samples=config.max_train_samples,
    )
    val_ds = SentimentDataset(
        val_csv,
        tokenizer,
        max_length=config.max_length,
        max_samples=config.max_val_samples,
    )
    print(f'样本数 train={len(train_ds)} val={len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # 准备2：模型
    my_model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config.num_labels,
    ).to(config.device)
    ln = my_model.bert.embeddings.LayerNorm.weight.detach().abs().mean().item()
    print(f'LayerNorm.weight mean(abs)={ln:.4f}（正常约 0.5~1.0；若接近 0.02 说明预训练未加载好）')

    # 准备3：损失（模型内部 CrossEntropy）
    # 准备4：优化器 + 调度器
    optimizer = AdamW(
        my_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    # 混合精度加速（4060）
    use_fp16 = bool(config.use_fp16 and config.device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    print(f'fp16={use_fp16}')

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    bad_epochs = 0
    history = []

    #  2个遍历：epoch / batch
    print('开始训练了...')
    for epoch in range(1, config.epochs + 1):
        my_model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f'第{epoch}/{config.epochs}轮')

        for batch in pbar:
            batch = {k: v.to(config.device) for k, v in batch.items()}
            #  5个核心：前向 → 损失 → 清零 → 反向 → 更新
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = my_model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(my_model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += float(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        train_loss = running / max(len(train_loader), 1)
        val_metrics = evaluate(my_model, val_loader, config.device)
        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
        }
        history.append(record)
        print(
            f'第{epoch}轮，训练损失：{train_loss:.4f}，'
            f'验证准确率：{val_metrics["accuracy"]:.4f}，'
            f'精确率：{val_metrics["precision"]:.4f}，'
            f'召回率：{val_metrics["recall"]:.4f}，'
            f'验证F1：{val_metrics["f1"]:.4f}'
        )

        #  1个保存（按验证 F1 存最佳）
        if val_metrics['f1'] > best_f1 + 1e-4:
            best_f1 = val_metrics['f1']
            bad_epochs = 0
            my_model.save_pretrained(out_dir / 'best')
            tokenizer.save_pretrained(out_dir / 'best')
            print(f'  已保存最佳模型到：{out_dir / "best"}，best_f1={best_f1:.4f}')
        else:
            bad_epochs += 1
            print(f'  验证 F1 未提升，patience {bad_epochs}/{config.patience}')
            if bad_epochs >= config.patience:
                print('早停：验证 F1 连续未提升')
                break

    metrics_path = Path(config.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump({'history': history, 'best_val_f1': best_f1}, f, ensure_ascii=False, indent=2)

    print(f'训练完成，最佳验证 F1：{best_f1:.4f}，指标写入：{metrics_path}')
    print('验收目标：test Acc>=0.92, F1>=0.90（请再跑 evaluate.py）')


if __name__ == '__main__':
    model_train()

"""
BiLSTM + 多头注意力 模型训练 (集成 Optuna 自动搜参)
只保存：权重 + 对比指标JSON + 路演图
"""

import os
import json
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import optuna  # 新增导入 Optuna

matplotlib.use('Agg')  # 非交互式后端

# ========== 🔑 修复中文字体乱码 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from bilstm_config import Config
from bilstm_model import BiLSTM_MultiHead
from bilstm_data_build_batch import build_batch_data

# 实例化配置文件
config = Config()

# ================================================================
# 工具函数 (与您的原代码完全一致)
# ================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for batch in tqdm(dataloader, desc="训练"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        logits, _ = model(input_ids)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='weighted')


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            logits, _ = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='weighted')


def plot_compare(baseline_f1, cnn_f1, bilstm_f1, save_path):
    """画三方对比图，标注+2% F1"""
    models = ['Baseline', 'CNN', 'BiLSTM']
    f1_scores = [baseline_f1, cnn_f1, bilstm_f1]
    gain = bilstm_f1 - cnn_f1

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.bar(models, f1_scores, color=['#E74C3C', '#3498DB', '#2ECC71'],
                  alpha=0.85, edgecolor='white', linewidth=2, width=0.6)

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2., score + 0.015,
                f'{score:.4f}', ha='center', va='bottom',
                fontsize=15, fontweight='bold', color='#2C3E50')

    if gain >= 0.02:
        ax.annotate(f'🔥 +{gain * 100:.1f}%',
                    xy=(2, bilstm_f1), xytext=(2, bilstm_f1 + 0.07),
                    ha='center', fontsize=15, fontweight='bold', color='#E74C3C',
                    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.5))

    if gain >= 0.02:
        ax.text(0.02, 0.96, '✅ 达标 (+2%)', transform=ax.transAxes,
                fontsize=15, fontweight='bold', color='#27AE60')
    else:
        ax.text(0.02, 0.96, f'❌ 未达标 (+{gain * 100:.1f}%)', transform=ax.transAxes,
                fontsize=15, fontweight='bold', color='#E74C3C')

    threshold = cnn_f1 + 0.02
    ax.axhline(y=threshold, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    ax.set_title('模型对比: Baseline vs CNN vs BiLSTM', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', alpha=0.85, label='Baseline'),
        Patch(facecolor='#3498DB', alpha=0.85, label='CNN'),
        Patch(facecolor='#2ECC71', alpha=0.85, label='BiLSTM'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 路演图已保存: {save_path}")


# ================================================================
# 【新增】Optuna 目标函数：单次搜索尝试
# ================================================================
def objective(trial):
    # 1. 从基础配置加载
    config = Config()
    device = config.device

    # 2. 定义超参数搜索空间（您可在此增加或修改参数）
    embed_dim = trial.suggest_categorical('embed_dim', [128, 256, 512])
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.3, 0.6)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # 3. 将 Optuna 搜索的参数写入 config 对象中
    config.embed_dim = embed_dim
    config.hidden_dim = hidden_dim
    config.num_layers = num_layers
    config.dropout = dropout
    config.learning_rate = learning_rate
    config.batch_size = batch_size

    # 打印当前试验参数（便于调试）
    print(f"🔍 [Trial {trial.number}] 参数: {config.embed_dim=}, {config.hidden_dim=}, {config.num_layers=}, {config.dropout=}, {config.learning_rate=}, {config.batch_size=}")

    # 4. 加载数据
    train_loader, val_loader, test_loader, vocab = build_batch_data(
        data_dir=config.data_process_path,
        vocab_path=config.save_vocab_path + "/vocab.json",
        max_vocab=config.vocab_size,
        max_len=config.max_len,
        batch_size=config.batch_size
    )

    # 5. 创建模型与优化器
    model = BiLSTM_MultiHead(
        vocab_size=len(vocab),
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pad_idx=vocab.get('<PAD>', 0)
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7)

    # 6. 训练与早停（相对精简，只求找到最优验证集F1）
    best_val_f1 = 0.0
    patience_counter = 0
    for epoch in range(1, config.ecope + 1):
        train_loss, _, _ = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{config.ecope} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # 在搜索过程中保存最优模型（可选）
            # torch.save(model.state_dict(), f'trial_{trial.number}_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    return best_val_f1  # Optuna 最大化该返回值


# ================================================================
# 【新增】最终带最优参数完整训练与保存
# ================================================================
def final_run(best_params):
    print("\n🏆 开始使用最优超参数进行完整训练...")
    device = config.device

    # 注入最佳参数
    for k, v in best_params.items():
        setattr(config, k, v)
    print(f"最终参数: {best_params}")

    # 加载数据
    train_loader, val_loader, test_loader, vocab = build_batch_data(
        data_dir=config.data_process_path,
        vocab_path=config.save_vocab_path + "/vocab.json",
        max_vocab=config.vocab_size,
        max_len=config.max_len,
        batch_size=config.batch_size
    )

    # 创建最终模型
    model = BiLSTM_MultiHead(
        vocab_size=len(vocab),
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pad_idx=vocab.get('<PAD>', 0)
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7)

    best_val_f1 = 0.0
    best_test_f1 = 0.0
    best_test_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, config.ecope + 1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch} | Train: {train_f1:.4f} | Val: {val_f1:.4f} | Test: {test_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_test_f1 = test_f1
            best_test_acc = test_acc
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

        scheduler.step(val_f1)

    # 加载最优模型并执行完整的保存流程
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("\n=== 最终测试 ===")
    _, final_test_acc, final_test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"测试集: Acc={final_test_acc:.4f}, F1={final_test_f1:.4f}")

    # --- 保存逻辑（与原代码一致）---
    # 1. 保存权重
    weight_dir = config.save_model_path
    os.makedirs(weight_dir, exist_ok=True)
    weight_path = os.path.join(weight_dir, 'bilstm_sentiment_best.pt')
    torch.save(model.state_dict(), weight_path)
    print(f"✅ 权重已保存: {weight_path}")

    # 2. 保存指标JSON
    cnn_metrics = {'f1': 0.86, 'accuracy': 0.86}
    try:
        with open('../../../artifacts/metrics/cnn_metrics.json', 'r') as f:
            cnn_metrics = json.load(f)
    except:
        print("⚠️ 未找到CNN指标，使用默认值")

    baseline_f1 = 0.85
    try:
        with open('../../../artifacts/metrics/baseline_metrics.json', 'r') as f:
            baseline_f1 = json.load(f).get('f1', 0.85)
    except:
        print("⚠️ 未找到Baseline指标，使用默认值")

    compare_data = {
        "baseline": {"f1": baseline_f1},
        "cnn": {"f1": cnn_metrics['f1']},
        "bilstm": {"f1": best_val_f1, "test_f1": final_test_f1, "test_acc": final_test_acc},
        "improvement": {
            "f1_gain": best_val_f1 - cnn_metrics['f1'],
            "f1_gain_percent": ((best_val_f1 - cnn_metrics['f1']) / cnn_metrics['f1']) * 100,
            "meets_2_percent": (best_val_f1 - cnn_metrics['f1']) >= 0.02
        }
    }

    metric_dir = config.metric_data_path
    os.makedirs(metric_dir, exist_ok=True)
    compare_path = os.path.join(metric_dir, 'bilstm_sentiment_metrics.json')
    with open(compare_path, 'w', encoding='utf-8') as f:
        json.dump(compare_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 对比指标已保存: {compare_path}")

    # 3. 路演图
    roadshow_dir = config.roadshow_data_path
    os.makedirs(roadshow_dir, exist_ok=True)
    plot_path = os.path.join(roadshow_dir, 'baseline_cnn_bilstm.png')
    plot_compare(baseline_f1, cnn_metrics['f1'], best_val_f1, plot_path)


def save_best_params_to_json(best_params, save_path):
    """
    将最优参数字典保存为 JSON 文件。

    Args:
        best_params (dict): Optuna 搜索到的最优参数字典
        save_path (str): JSON 文件的保存路径
    """
    # 1. 确保文件夹存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 2. 写入 JSON
    with open(save_path + '/best_params.json', 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4, ensure_ascii=False)

    print(f"✅ 最优参数已保存至 JSON: {save_path + '/best_params.json'}")


# ================================================================
# 主程序
# ================================================================
if __name__ == "__main__":
    print("\n🚀 启动 Optuna 自动超参数搜索...")

    # 创建 Optuna 研究实例，目标为最大化 F1
    study = optuna.create_study(direction='maximize')

    # 启动 20 次搜索（若电脑性能较好可增加 n_trials，如 30-50）
    study.optimize(objective, n_trials=10, show_progress_bar=True, n_jobs=-1)

    print("\n🎯 搜索完成！最优参数:")
    print(study.best_params)
    print(f"最佳验证 F1: {study.best_value:.4f}")

    # 使用最优参数执行最终的保存流程
    final_run(study.best_params)

    save_best_params_to_json(study.best_params, config.save_best_param_path)

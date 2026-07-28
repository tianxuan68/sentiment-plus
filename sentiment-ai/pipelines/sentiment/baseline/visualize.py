"""
结果可视化模块
生成路演所需的图表
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from config import *

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
plt.style.use('seaborn-v0_8-darkgrid')


def plot_comparison_chart(results_df, save_path=None):
    """
    绘制模型对比柱状图（路演用）

    Args:
        results_df: 结果DataFrame
        save_path: 保存路径
    """
    # 准备数据
    models = results_df['model'].tolist()
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 7))

    # 设置柱状图位置
    x = np.arange(len(models))
    width = 0.2
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#70AD47']

    # 绘制每个指标的柱状图
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = results_df[metric].values
        bars = ax.bar(x + i*width, values, width, label=label, color=color, edgecolor='white', linewidth=1.5)

        # 在柱子上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 设置图表属性
    ax.set_xlabel('模型', fontsize=14, fontweight='bold')
    ax.set_ylabel('分数', fontsize=14, fontweight='bold')
    ax.set_title('Baseline模型性能对比（测试集）', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加验收阈值线（85% Acc / 0.85 F1）
    ax.axhline(y=0.85, color='red', linestyle='--', linewidth=2, alpha=0.7, label='85% 阈值')

    # 添加最佳F1标注
    best_idx = results_df['f1'].idxmax()
    best_model = results_df.loc[best_idx, 'model']
    best_f1 = results_df.loc[best_idx, 'f1']
    ax.annotate(f'最佳F1: {best_f1:.3f}\n({best_model})',
                xy=(best_idx + width*3, best_f1),
                xytext=(best_idx + width*3, best_f1 + 0.1),
                ha='center',
                fontsize=10,
                fontweight='bold',
                color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    plt.tight_layout()

    # 保存图表
    if save_path is None:
        save_path = os.path.join(RESULT_DIR, 'comparison_chart.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存至: {save_path}")

    plt.close()


def plot_metrics_table(results_df, save_path=None):
    """
    生成指标表格图片（路演用）

    Args:
        results_df: 结果DataFrame
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    # 格式化数值
    display_df = results_df.copy()
    for col in ['accuracy', 'precision', 'recall', 'f1']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

    # 重命名列
    display_df.columns = ['模型', 'Accuracy', 'Precision', 'Recall', 'F1-Score']

    # 创建表格
    table = ax.table(cellText=display_df.values,
                    colLabels=display_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # 设置表头样式
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置行颜色
    for i in range(1, len(display_df) + 1):
        if i % 2 == 0:
            for j in range(len(display_df.columns)):
                table[(i, j)].set_facecolor('#E7E6E6')

    # 高亮最佳F1行
    best_idx = results_df['f1'].idxmax() + 1
    for j in range(len(display_df.columns)):
        table[(best_idx, j)].set_facecolor('#C6EFCE')
        table[(best_idx, j)].set_text_props(weight='bold')

    plt.title('Baseline模型性能指标表', fontsize=14, fontweight='bold', pad=20)

    if save_path is None:
        save_path = os.path.join(RESULT_DIR, 'metrics_table.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"指标表格已保存至: {save_path}")

    plt.close()


def generate_metrics_text(results_df):
    """
    生成指标文本文件

    Args:
        results_df: 结果DataFrame
    """
    # 格式化数值
    formatted_df = results_df.copy()
    for col in ['accuracy', 'precision', 'recall', 'f1']:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")

    # 保存表格
    table_path = os.path.join(RESULT_DIR, 'metrics_table.txt')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Baseline模型性能对比表\n")
        f.write("=" * 60 + "\n\n")
        f.write(formatted_df.to_string(index=False))
        f.write("\n\n" + "=" * 60 + "\n")

        # 添加最佳模型信息
        best_idx = results_df['f1'].idxmax()
        best_model = results_df.loc[best_idx]
        f.write(f"最佳模型: {best_model['model']}\n")
        f.write(f"Accuracy:  {best_model['accuracy']:.4f}\n")
        f.write(f"Precision: {best_model['precision']:.4f}\n")
        f.write(f"Recall:    {best_model['recall']:.4f}\n")
        f.write(f"F1-Score:  {best_model['f1']:.4f}\n")
        f.write("=" * 60 + "\n")

    print(f"指标文本已保存至: {table_path}")


def main(results_df=None):
    """
    主可视化流程

    Args:
        results_df: 结果DataFrame，如果为None则自动加载
    """
    print("=" * 60)
    print("生成可视化结果")
    print("=" * 60)

    # 如果未提供结果，尝试加载
    if results_df is None:
        results_path = os.path.join(RESULT_DIR, 'metrics.csv')
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
        else:
            print("错误: 未找到评估结果，请先运行 evaluate.py")
            return

    # 确保结果目录存在
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 生成对比图
    print("\n生成对比柱状图...")
    plot_comparison_chart(results_df)

    # 生成指标表格图
    print("\n生成指标表格图...")
    plot_metrics_table(results_df)

    # 生成指标文本
    print("\n生成指标文本...")
    generate_metrics_text(results_df)

    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)
    print(f"输出文件:")
    print(f"  - 对比图: {os.path.join(RESULT_DIR, 'comparison_chart.png')}")
    print(f"  - 表格图: {os.path.join(RESULT_DIR, 'metrics_table.png')}")
    print(f"  - 指标文本: {os.path.join(RESULT_DIR, 'metrics_table.txt')}")


if __name__ == '__main__':
    main()

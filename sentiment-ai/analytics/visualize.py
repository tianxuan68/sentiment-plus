"""
可视化模块
生成饼图、评分柱状图、KPI 卡图片（路演用）
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from config import *

# 设置中文字体
plt.rcParams['font.sans-serif'] = FONT_SANS_SERIF
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')


def plot_sentiment_pie(sentiment_stats, save_path=None):
    """
    绘制情感正负比例饼图

    Args:
        sentiment_stats: 正负比例统计结果
        save_path: 保存路径
    """
    labels = ['负向（不满意）', '正向（满意）']
    sizes = [sentiment_stats['negative'], sentiment_stats['positive']]
    colors = COLORS_PIE
    explode = (0, 0.05)  # 突出正向

    fig, ax = plt.subplots(figsize=FIG_SIZE_PIE)
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 13}
    )

    # 设置标签样式
    for text in texts:
        text.set_fontsize(13)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    # 标题
    ax.set_title(
        f"用户情感正负比例\n（总样本: {sentiment_stats['total']} 条）",
        fontsize=16, fontweight='bold', pad=20
    )

    # 添加正负比标注
    ax.text(0, -1.35, f"正负比: {sentiment_stats['ratio']}",
            ha='center', fontsize=12, color='#555555')

    if save_path is None:
        save_path = os.path.join(CHART_DIR, 'sentiment_pie.png')

    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"正负比例饼图已保存至: {save_path}")
    plt.close()


def plot_score_bar(score_stats, save_path=None):
    """
    绘制评分分布柱状图

    Args:
        score_stats: 评分分布统计结果
        save_path: 保存路径
    """
    scores = list(score_stats['distribution'].keys())
    counts = [score_stats['distribution'][s]['count'] for s in scores]
    pcts = [score_stats['distribution'][s]['percentage'] for s in scores]

    # 星级标签
    star_labels = [f"{s}分\n{'★'*(s+1)}{'☆'*(4-s)}" for s in scores]

    fig, ax = plt.subplots(figsize=FIG_SIZE_BAR)

    bars = ax.bar(star_labels, counts, color=COLORS_BAR,
                  edgecolor='white', linewidth=1.5, width=0.6)

    # 在柱子上添加数值和百分比
    for bar, count, pct in zip(bars, counts, pcts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.01,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('评分', fontsize=14, fontweight='bold')
    ax.set_ylabel('样本数', fontsize=14, fontweight='bold')
    ax.set_title(
        f"商品评分分布\n（总样本: {score_stats['total']} 条，平均评分: {score_stats['mean_score']:.2f}/4.00）",
        fontsize=16, fontweight='bold', pad=20
    )
    ax.set_ylim([0, max(counts) * 1.15])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    if save_path is None:
        save_path = os.path.join(CHART_DIR, 'score_distribution.png')

    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"评分分布柱状图已保存至: {save_path}")
    plt.close()


def plot_kpi_cards(report, save_path=None):
    """
    生成 KPI 指标卡片图

    Args:
        report: 满意度报告
        save_path: 保存路径
    """
    summary = report['summary']
    sentiment = report['sentiment_ratio']
    score = report['score_distribution']

    fig, axes = plt.subplots(1, 4, figsize=FIG_SIZE_KPI)
    fig.suptitle('用户满意度 KPI 指标', fontsize=16, fontweight='bold', y=1.02)

    # KPI 1: 中文总样本
    ax = axes[0]
    ax.axis('off')
    ax.text(0.5, 0.65, f"{summary['cn_total']:,}", ha='center', va='center',
            fontsize=32, fontweight='bold', color=COLORS_KPI[0])
    ax.text(0.5, 0.25, '中文评论总数', ha='center', va='center',
            fontsize=13, color='#555555')

    # KPI 2: 中文满意度
    ax = axes[1]
    ax.axis('off')
    ax.text(0.5, 0.65, f"{summary['cn_satisfaction_rate']:.1f}%", ha='center', va='center',
            fontsize=32, fontweight='bold', color=COLORS_KPI[1])
    ax.text(0.5, 0.25, f"中文满意度\n（正负比 {sentiment['ratio']}）",
            ha='center', va='center', fontsize=13, color='#555555')

    # KPI 3: 英文平均评分
    ax = axes[2]
    ax.axis('off')
    ax.text(0.5, 0.65, f"{summary['en_mean_score']:.2f}", ha='center', va='center',
            fontsize=32, fontweight='bold', color=COLORS_KPI[2])
    ax.text(0.5, 0.25, '英文平均评分\n（满分 4.00）',
            ha='center', va='center', fontsize=13, color='#555555')

    # KPI 4: 英文满意度
    ax = axes[3]
    ax.axis('off')
    ax.text(0.5, 0.65, f"{summary['en_satisfaction_rate']:.1f}%", ha='center', va='center',
            fontsize=32, fontweight='bold', color=COLORS_KPI[3])
    ax.text(0.5, 0.25, f"英文满意度\n（≥3分, 共{score['satisfied_count']:,}条）",
            ha='center', va='center', fontsize=13, color='#555555')

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(CHART_DIR, 'kpi_cards.png')

    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"KPI 卡片图已保存至: {save_path}")
    plt.close()


def main(report=None):
    """
    主可视化流程

    Args:
        report: 满意度报告，如果为None则自动加载
    """
    print("=" * 60)
    print("生成可视化结果")
    print("=" * 60)

    # 如果未提供报告，尝试加载
    if report is None:
        import json
        report_path = os.path.join(REPORT_DIR, 'satisfaction_report.json')
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
        else:
            print("错误: 未找到满意度报告，请先运行 stats.py")
            return

    os.makedirs(CHART_DIR, exist_ok=True)

    print("\n生成情感正负比例饼图...")
    plot_sentiment_pie(report['sentiment_ratio'])

    print("\n生成评分分布柱状图...")
    plot_score_bar(report['score_distribution'])

    print("\n生成 KPI 指标卡片图...")
    plot_kpi_cards(report)

    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)
    print(f"输出文件:")
    print(f"  - 饼图: {os.path.join(CHART_DIR, 'sentiment_pie.png')}")
    print(f"  - 柱状图: {os.path.join(CHART_DIR, 'score_distribution.png')}")
    print(f"  - KPI卡: {os.path.join(CHART_DIR, 'kpi_cards.png')}")


if __name__ == '__main__':
    main()

"""
核心统计模块
计算正负比例、评分分布、满意度指标
"""
import pandas as pd
import numpy as np
import json
import os
from config import *


def calc_sentiment_ratio(cn_df):
    """
    计算中文情感数据的正负比例

    Args:
        cn_df: 中文数据DataFrame，包含 label 列

    Returns:
        dict: 正负比例统计结果
    """
    total = len(cn_df)
    pos_count = int(cn_df[CN_LABEL_COLUMN].sum())
    neg_count = total - pos_count

    pos_pct = pos_count / total * 100 if total > 0 else 0
    neg_pct = neg_count / total * 100 if total > 0 else 0
    ratio = f"{pos_count}:{neg_count}" if neg_count > 0 else f"{pos_count}:0"

    result = {
        'total': total,
        'positive': pos_count,
        'negative': neg_count,
        'positive_pct': round(pos_pct, 2),
        'negative_pct': round(neg_pct, 2),
        'ratio': ratio,
        'satisfaction_rate': round(pos_pct, 2)  # 满意度=正向率
    }

    print(f"\n{'='*50}")
    print("情感正负比例统计")
    print(f"{'='*50}")
    print(f"总样本数: {total}")
    print(f"正向(满意): {pos_count} ({pos_pct:.2f}%)")
    print(f"负向(不满意): {neg_count} ({neg_pct:.2f}%)")
    print(f"正负比: {ratio}")
    print(f"满意度: {pos_pct:.2f}%")

    return result


def calc_score_distribution(en_df):
    """
    计算英文评分数据的评分分布

    Args:
        en_df: 英文数据DataFrame，包含 overall 列

    Returns:
        dict: 评分分布统计结果
    """
    scores = en_df[EN_SCORE_COLUMN].astype(int)
    total = len(scores)
    mean_score = float(scores.mean())

    # 各分数段统计（0~4）
    distribution = {}
    for score in range(5):
        count = int((scores == score).sum())
        pct = count / total * 100 if total > 0 else 0
        distribution[score] = {
            'count': count,
            'percentage': round(pct, 2)
        }

    # 满意度：评分>=3 视为满意
    satisfied = int((scores >= 3).sum())
    satisfaction_rate = satisfied / total * 100 if total > 0 else 0

    result = {
        'total': total,
        'mean_score': round(mean_score, 2),
        'distribution': distribution,
        'satisfied_count': satisfied,
        'satisfaction_rate': round(satisfaction_rate, 2)
    }

    print(f"\n{'='*50}")
    print("评分分布统计")
    print(f"{'='*50}")
    print(f"总样本数: {total}")
    print(f"平均评分: {mean_score:.2f} / 4.00")
    print(f"\n各分数段分布:")
    for score, info in distribution.items():
        stars = '★' * (score + 1) + '☆' * (4 - score)
        print(f"  {score}分 ({stars}): {info['count']} 条 ({info['percentage']:.2f}%)")
    print(f"\n满意度(≥3分): {satisfied} ({satisfaction_rate:.2f}%)")

    return result


def generate_report(sentiment_stats, score_stats):
    """
    汇总生成满意度报告

    Args:
        sentiment_stats: 正负比例统计结果
        score_stats: 评分分布统计结果

    Returns:
        dict: 完整的满意度报告
    """
    report = {
        'summary': {
            'cn_total': sentiment_stats['total'],
            'en_total': score_stats['total'],
            'cn_satisfaction_rate': sentiment_stats['satisfaction_rate'],
            'en_satisfaction_rate': score_stats['satisfaction_rate'],
            'en_mean_score': score_stats['mean_score']
        },
        'sentiment_ratio': sentiment_stats,
        'score_distribution': score_stats
    }

    # 保存 JSON 报告
    report_path = os.path.join(REPORT_DIR, 'satisfaction_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n满意度报告已保存至: {report_path}")

    # 保存 CSV 汇总
    summary_data = [
        {'指标': '中文总样本', '值': sentiment_stats['total']},
        {'指标': '中文正向数', '值': sentiment_stats['positive']},
        {'指标': '中文负向数', '值': sentiment_stats['negative']},
        {'指标': '中文满意度(%)', '值': sentiment_stats['satisfaction_rate']},
        {'指标': '中文正负比', '值': sentiment_stats['ratio']},
        {'指标': '英文总样本', '值': score_stats['total']},
        {'指标': '英文平均评分', '值': score_stats['mean_score']},
        {'指标': '英文满意度(≥3分)(%)', '值': score_stats['satisfaction_rate']}
    ]
    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(REPORT_DIR, 'satisfaction_summary.csv')
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"汇总CSV已保存至: {csv_path}")

    # 保存评分分布表
    dist_rows = []
    for score, info in score_stats['distribution'].items():
        dist_rows.append({
            '评分': score,
            '样本数': info['count'],
            '占比(%)': info['percentage']
        })
    dist_df = pd.DataFrame(dist_rows)
    dist_path = os.path.join(REPORT_DIR, 'score_distribution.csv')
    dist_df.to_csv(dist_path, index=False, encoding='utf-8-sig')
    print(f"评分分布表已保存至: {dist_path}")

    return report


def main(cn_df=None, en_df=None):
    """
    主统计流程

    Args:
        cn_df: 中文数据DataFrame，如果为None则自动加载
        en_df: 英文数据DataFrame，如果为None则自动加载

    Returns:
        dict: 满意度报告
    """
    print("=" * 60)
    print("满意度统计分析")
    print("=" * 60)

    # 按需加载数据
    if cn_df is None:
        from data_loader import load_chinese_data
        cn_df = load_chinese_data()

    if en_df is None:
        from data_loader import load_english_data
        en_df = load_english_data()

    # 计算统计
    sentiment_stats = calc_sentiment_ratio(cn_df)
    score_stats = calc_score_distribution(en_df)

    # 生成报告
    report = generate_report(sentiment_stats, score_stats)

    print("\n" + "=" * 60)
    print("满意度统计完成！")
    print("=" * 60)
    print(f"中文满意度: {report['summary']['cn_satisfaction_rate']:.2f}%")
    print(f"英文满意度: {report['summary']['en_satisfaction_rate']:.2f}%")
    print(f"英文平均评分: {report['summary']['en_mean_score']:.2f} / 4.00")
    print("=" * 60)

    return report


if __name__ == '__main__':
    main()

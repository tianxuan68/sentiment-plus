"""
满意度统计完整流程主入口
运行此脚本完成：数据加载 -> 统计计算 -> 报告生成 -> 可视化

使用方法:
    python run_analytics.py

可选参数:
    --cn-data: 指定中文数据文件路径
    --en-data: 指定英文数据文件路径
    --output: 指定输出目录
    --skip-charts: 跳过可视化步骤
"""
import os
import sys
import argparse
from config import *
from data_loader import load_chinese_data, load_english_data
from stats import calc_sentiment_ratio, calc_score_distribution, generate_report
from visualize import main as visualize_main


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='满意度统计分析')
    parser.add_argument('--cn-data', type=str, default=None,
                       help='指定中文数据文件路径')
    parser.add_argument('--en-data', type=str, default=None,
                       help='指定英文数据文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='指定输出目录')
    parser.add_argument('--skip-charts', action='store_true',
                       help='跳过可视化步骤')
    return parser.parse_args()


def main():
    """
    满意度统计完整流程
    """
    args = parse_args()

    print("=" * 70)
    print("商品评论满意度统计分析")
    print("=" * 70)
    print(f"工作目录: {BASE_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 70)

    try:
        # 步骤1: 数据加载
        print("\n[步骤 1/4] 数据加载")
        print("-" * 50)

        cn_df = None
        en_df = None

        try:
            cn_df = load_chinese_data(args.cn_data)
        except FileNotFoundError as e:
            print(f"警告: {e}")

        try:
            en_df = load_english_data(args.en_data)
        except FileNotFoundError as e:
            print(f"警告: {e}")

        if cn_df is None and en_df is None:
            print("\n错误: 未找到任何数据文件，无法继续分析")
            print(f"请确保数据文件已放置在: {DATA_DIR}")
            print(f"  - 中文数据: {os.path.join(DATA_DIR, CN_DATA_FILE)}")
            print(f"  - 英文数据: {os.path.join(DATA_DIR, EN_DATA_FILE)}")
            return None

        # 步骤2: 统计计算
        print("\n[步骤 2/4] 统计计算")
        print("-" * 50)

        sentiment_stats = None
        score_stats = None

        if cn_df is not None:
            sentiment_stats = calc_sentiment_ratio(cn_df)

        if en_df is not None:
            score_stats = calc_score_distribution(en_df)

        # 步骤3: 报告生成
        print("\n[步骤 3/4] 报告生成")
        print("-" * 50)

        # 如果某个数据源缺失，用空值占位
        if sentiment_stats is None:
            sentiment_stats = {
                'total': 0, 'positive': 0, 'negative': 0,
                'positive_pct': 0, 'negative_pct': 0,
                'ratio': 'N/A', 'satisfaction_rate': 0
            }
        if score_stats is None:
            score_stats = {
                'total': 0, 'mean_score': 0,
                'distribution': {i: {'count': 0, 'percentage': 0} for i in range(5)},
                'satisfied_count': 0, 'satisfaction_rate': 0
            }

        report = generate_report(sentiment_stats, score_stats)

        # 步骤4: 可视化
        if not args.skip_charts:
            print("\n[步骤 4/4] 结果可视化")
            print("-" * 50)
            visualize_main(report)
        else:
            print("\n[步骤 4/4] 跳过可视化（--skip-charts）")

        # 验收标准检查
        print("\n" + "=" * 70)
        print("验收标准检查")
        print("=" * 70)

        # 检查正负比例
        has_ratio = sentiment_stats['total'] > 0
        print(f"{'[PASS]' if has_ratio else '[FAIL]'} 正负比例: "
              f"{sentiment_stats.get('ratio', 'N/A')} "
              f"({sentiment_stats.get('positive', 0)}:{sentiment_stats.get('negative', 0)})")

        # 检查评分分布
        has_dist = score_stats['total'] > 0
        print(f"{'[PASS]' if has_dist else '[FAIL]'} 评分分布: "
              f"{'已生成' if has_dist else '未生成'}")

        # 检查报告文件
        report_path = os.path.join(REPORT_DIR, 'satisfaction_report.json')
        has_report = os.path.exists(report_path)
        print(f"{'[PASS]' if has_report else '[FAIL]'} 满意度报告: "
              f"{'已导出' if has_report else '未导出'}")

        # 检查图表
        if not args.skip_charts:
            chart_files = ['sentiment_pie.png', 'score_distribution.png', 'kpi_cards.png']
            for f in chart_files:
                path = os.path.join(CHART_DIR, f)
                exists = os.path.exists(path)
                print(f"{'[PASS]' if exists else '[FAIL]'} 图表 {f}: "
                      f"{'已生成' if exists else '未生成'}")

        print("=" * 70)

        # 最终总结
        print("\n" + "=" * 70)
        print("满意度统计分析完成！")
        print("=" * 70)
        print(f"\n结果保存在:")
        print(f"  报告文件: {REPORT_DIR}")
        print(f"  图表文件: {CHART_DIR}")

        if sentiment_stats['total'] > 0:
            print(f"\n中文满意度: {sentiment_stats['satisfaction_rate']:.2f}%")
        if score_stats['total'] > 0:
            print(f"英文满意度: {score_stats['satisfaction_rate']:.2f}%")
            print(f"英文平均评分: {score_stats['mean_score']:.2f} / 4.00")

        return report

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    report = main()

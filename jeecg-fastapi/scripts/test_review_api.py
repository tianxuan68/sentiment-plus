"""
评论分析 Mock 与聚合逻辑单元测试（无需数据库）

用法（在 jeecg-fastapi 目录）:
    python scripts/test_review_api.py
"""
# 1.导包
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services import sentiment_ai_client as ai


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(ROOT).replace('\\', '/') + '/'
        self.positive_text = '物流很快，质量不错，客服耐心'
        self.negative_text = '太差了，包装简陋，失望退货'
        self.batch_texts = ['物流很快', '质量太差了']
        self.keyword_text = '物流很快，包装简陋，性价比还可以'
        self.entity_text = '物流很快，但是包装有点简陋'
        self.top_n = 5
        self.batch_top_n = 3


config = Config()


def test_mock_analyze_positive():
    # 3.正向情感分析
    result = asyncio.run(ai.analyze_review(config.positive_text, top_n=config.top_n))
    assert result.sentiment.label == 1
    assert result.sentiment.label_text == '正向'
    assert len(result.keywords) >= 1
    assert result.source == 'mock'
    assert result.latency_ms is not None
    assert result.text_preview


def test_mock_analyze_negative():
    # 4.负向情感分析
    result = asyncio.run(ai.analyze_review(config.negative_text, top_n=config.top_n))
    assert result.sentiment.label == 0
    assert any(a.aspect == '包装' for a in result.aspects)


def test_batch_analyze():
    # 5.批量分析
    batch = asyncio.run(ai.analyze_batch(config.batch_texts, top_n=config.batch_top_n))
    assert batch.total == 2
    assert len(batch.items) == 2
    assert batch.avg_latency_ms is not None
    labels = {item.sentiment.label for item in batch.items}
    assert 0 in labels and 1 in labels


def test_polarity_and_trend():
    # 6.极性与趋势统计
    polarity = asyncio.run(ai.get_polarity_stats())
    assert polarity.total == polarity.positive + polarity.negative
    assert 0 < polarity.positive_ratio < 1

    trend = asyncio.run(ai.get_trend_stats())
    assert len(trend) >= 5
    assert trend[0].positive is not None


def test_dashboard_summary():
    # 7.仪表盘摘要
    summary = asyncio.run(ai.get_dashboard_summary())
    assert 0 <= summary.satisfaction_score <= 100
    assert summary.trend_direction in ('up', 'down', 'stable')


def test_pros_cons():
    # 8.优缺点列表
    data = asyncio.run(ai.get_pros_cons())
    assert len(data.pros) == 10
    assert len(data.cons) == 10


def test_ai_status_mock_mode():
    # 9.AI 状态（mock 模式）
    status = asyncio.run(ai.get_ai_status())
    assert status.mode == 'mock'
    assert status.mock_enabled is True


def test_model_compare():
    # 10.模型对比
    data = asyncio.run(ai.get_model_compare())
    assert len(data.metrics) == 3
    assert data.bilstm_vs_baseline_f1_gain >= 0.02
    assert data.best_model == 'BERT Fine-tune'


def test_keyword_compare():
    # 11.关键词对比
    data = asyncio.run(ai.compare_keywords(config.keyword_text, top_n=config.top_n))
    assert len(data.keybert) >= 1
    assert len(data.tfidf) >= 1


def test_rating_distribution():
    # 12.评分分布
    data = asyncio.run(ai.get_rating_distribution())
    assert data.total == sum(b.count for b in data.buckets)
    assert len(data.buckets) == 5


def test_analyze_entities():
    # 13.实体分析
    result = asyncio.run(ai.analyze_review(config.entity_text, top_n=config.top_n))
    assert len(result.entities) >= 1
    assert result.active_model == 'BERT'


def process_data():
    # 14.运行全部测试
    tests = [
        test_mock_analyze_positive,
        test_mock_analyze_negative,
        test_batch_analyze,
        test_polarity_and_trend,
        test_dashboard_summary,
        test_pros_cons,
        test_ai_status_mock_mode,
        test_model_compare,
        test_keyword_compare,
        test_rating_distribution,
        test_analyze_entities,
    ]
    for fn in tests:
        fn()
        print(f'OK {fn.__name__}')
    print(f'\n全部 {len(tests)} 项测试通过')


if __name__ == '__main__':
    process_data()

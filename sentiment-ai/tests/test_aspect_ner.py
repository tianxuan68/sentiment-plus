"""
5号 aspect_ner 单元测试（无需 GPU）
"""

# 1.导包
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.aspect_ner.annotate import annotate_bio, find_aspect_spans, load_annotation_pool
from pipelines.aspect_ner.constants import ASPECT_TYPES
from pipelines.aspect_ner.predict import predict_entities, predict_fun


def test_find_aspect_spans():
    # 场景1：多属性触发词
    text = '物流很快，但是包装有点简陋，性价比还可以'
    spans = find_aspect_spans(text)
    aspects = {s[2] for s in spans}
    assert '物流' in aspects
    assert '包装' in aspects
    print('test_find_aspect_spans 通过')


def test_annotate_bio():
    # 场景2：BIO + polarity
    rec = annotate_bio('1', '物流很快，包装简陋', sentence_label=1)
    assert rec['tokens']
    assert len(rec['labels']) == len(rec['tokens'])
    assert any(lab.startswith('B-') for lab in rec['labels'])
    assert rec['sentiment_rows']
    print('test_annotate_bio 通过')


def test_predict_entities_schema():
    # 场景3：推理字段契约
    entities = predict_entities('物流很快，包装简陋')
    assert entities
    ent = entities[0]
    for key in ('aspect', 'text', 'start', 'end', 'polarity', 'polarity_text'):
        assert key in ent
    assert ent['polarity'] in (0, 1)
    print('test_predict_entities_schema 通过')


def test_predict_fun_dict():
    # 场景4：szai6 predict_fun(dict) 写回 predict_class
    sample = {'text': '物流很快，包装简陋'}
    out = predict_fun(sample)
    assert out is sample
    assert 'entities' in out
    assert 'predict_class' in out
    assert out['predict_class'] == out['entities']
    print(f'predict_fun 实体数：{len(out["entities"])}')


def test_five_aspect_types():
    assert len(ASPECT_TYPES) >= 5
    print(f'属性类型数：{len(ASPECT_TYPES)}')


def test_pool_load_if_exists():
    pool_path = ROOT / 'data' / 'samples' / 'aspect_annotate_pool.csv'
    if pool_path.exists():
        pool = load_annotation_pool()
        assert len(pool) >= 100
        print(f'标注池条数：{len(pool)}')
    else:
        print('标注池不存在，跳过')


if __name__ == '__main__':
    tests = [
        test_find_aspect_spans,
        test_annotate_bio,
        test_predict_entities_schema,
        test_predict_fun_dict,
        test_five_aspect_types,
        test_pool_load_if_exists,
    ]
    for fn in tests:
        fn()
        print(f'OK  {fn.__name__}')
    print(f'全部 {len(tests)} 个测试通过')

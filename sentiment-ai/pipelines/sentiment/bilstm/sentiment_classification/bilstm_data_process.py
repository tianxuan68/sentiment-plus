"""
数据集处理模块
功能：加载中英文数据、划分训练/验证/测试集、分词、保存处理后的数据
"""
import os
import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from bilstm_config import Config
from bilstm_data_util import tokenize_text


config_obj = Config()


# ================================================================
# 1. 数据加载
# ================================================================

def load_data(path):
    """加载中文数据"""
    df = pd.read_csv(path)
    texts = df['sentence'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    return texts, labels

def load_chinese_data(path):
    """加载中文数据"""
    df = pd.read_csv(path)
    texts = df['text_cleaned'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()
    return texts, labels


import re


def clean_texts(texts, labels):
    # 如果传入的是单个字符串（兼容性处理），构建单元素列表
    if isinstance(texts, str):
        texts = [texts]
    if isinstance(labels, (int, float)):
        labels = [labels]

    # 检查两个列表长度是否一致
    if len(texts) != len(labels):
        raise ValueError(f"texts 和 labels 的长度不一致: {len(texts)} vs {len(labels)}")

    # 定义保留字符的正则
    allowed_chars = r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；：“”‘’《》【】（）、…—\.\,\!\?\;\:\'\"\-]'

    cleaned_texts = []
    cleaned_labels = []

    for text, label in zip(texts, labels):
        # 1. 强制转为字符串，防止 NaN(float) 报错
        text_str = str(text)

        # 2. 清洗文本
        cleaned = re.sub(allowed_chars, '', text_str)

        # 3. 去除首尾空格并判断是否为空
        # 如果是 'nan' 字符串，可以视作无效文本，strip后变成 'nan'，可以判断去掉
        if cleaned.strip() == '' or cleaned.strip().lower() == 'nan':
            continue  # 跳过此行，不添加到结果中

        # 4. 如果有效，同时添加清洗后的文本和对应的标签
        cleaned_texts.append(cleaned)
        cleaned_labels.append(int(label) if label != '' else label)  # 确保标签类型正确，视情况调整

    return cleaned_texts, cleaned_labels

def load_english_data(path, threshold=3):
    """加载英文数据，评分转二分类"""
    df = pd.read_csv(path)
    texts = df['reviewText'].astype(str).tolist()
    ratings = df['overall'].astype(float).tolist()
    labels = [1 if r >= threshold else 0 for r in ratings]
    return texts, labels


# ================================================================
# 2. 数据划分
# ================================================================

def split_data(texts, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=68):
    """划分训练集、验证集、测试集 (8:1:1)"""
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_ratio, random_state=random_seed, shuffle=True
    )

    val_in_train_ratio = val_ratio / (train_ratio + val_ratio)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=val_in_train_ratio,
        random_state=random_seed, shuffle=True
    )

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def split_mixed_data(zh_texts, zh_labels, en_texts, en_labels,
                     train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=68):
    """中英文数据分别按比例划分后组合"""
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = [], [], [], [], [], []

    if zh_texts:
        zh_train_t, zh_train_l, zh_val_t, zh_val_l, zh_test_t, zh_test_l = split_data(
            zh_texts, zh_labels, train_ratio, val_ratio, test_ratio, random_seed
        )
        train_texts.extend(zh_train_t)
        train_labels.extend(zh_train_l)
        val_texts.extend(zh_val_t)
        val_labels.extend(zh_val_l)
        test_texts.extend(zh_test_t)
        test_labels.extend(zh_test_l)
        print(f"  中文: 训练{len(zh_train_t)}, 验证{len(zh_val_t)}, 测试{len(zh_test_t)}")

    if en_texts:
        en_train_t, en_train_l, en_val_t, en_val_l, en_test_t, en_test_l = split_data(
            en_texts, en_labels, train_ratio, val_ratio, test_ratio, random_seed
        )
        train_texts.extend(en_train_t)
        train_labels.extend(en_train_l)
        val_texts.extend(en_val_t)
        val_labels.extend(en_val_l)
        test_texts.extend(en_test_t)
        test_labels.extend(en_test_l)
        print(f"  英文: 训练{len(en_train_t)}, 验证{len(en_val_t)}, 测试{len(en_test_t)}")

    import random
    combined = list(zip(train_texts, train_labels))
    random.seed(random_seed)
    random.shuffle(combined)
    if combined:
        train_texts, train_labels = zip(*combined)
        train_texts, train_labels = list(train_texts), list(train_labels)

    combined = list(zip(val_texts, val_labels))
    random.shuffle(combined)
    if combined:
        val_texts, val_labels = zip(*combined)
        val_texts, val_labels = list(val_texts), list(val_labels)

    combined = list(zip(test_texts, test_labels))
    random.shuffle(combined)
    if combined:
        test_texts, test_labels = zip(*combined)
        test_texts, test_labels = list(test_texts), list(test_labels)

    total = len(train_texts) + len(val_texts) + len(test_texts)
    if total > 0:
        print(f"\n✅ 划分完成: 训练{len(train_texts)}, 验证{len(val_texts)}, 测试{len(test_texts)}")

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


# ================================================================
# 3. 分词处理
# ================================================================

def tokenize_data(texts, zh_mode='char'):
    """对文本列表进行分词"""
    tokenized = []
    for text in texts:
        tokens = tokenize_text(text, zh_mode)
        tokenized.append(' '.join(tokens))
    return tokenized


# ================================================================
# 4. 保存数据集
# ================================================================

def save_dataset(train_texts, train_labels, val_texts, val_labels,
                 test_texts, test_labels, output_dir="./processed_data"):
    """保存处理后的数据集到CSV文件（已分词）"""
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame({'text': train_texts, 'label': train_labels}).to_csv(
        f"{output_dir}/train.csv", index=False, encoding='utf-8')
    pd.DataFrame({'text': val_texts, 'label': val_labels}).to_csv(
        f"{output_dir}/val.csv", index=False, encoding='utf-8')
    pd.DataFrame({'text': test_texts, 'label': test_labels}).to_csv(
        f"{output_dir}/test.csv", index=False, encoding='utf-8')

    stats = {
        'train_size': len(train_texts), 'val_size': len(val_texts), 'test_size': len(test_texts),
        'train_positive': sum(train_labels), 'train_negative': len(train_labels) - sum(train_labels),
        'val_positive': sum(val_labels), 'val_negative': len(val_labels) - sum(val_labels),
        'test_positive': sum(test_labels), 'test_negative': len(test_labels) - sum(test_labels),
    }
    with open(f"{output_dir}/stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 数据已保存: {output_dir}")
    print(f"   训练集: {len(train_texts)} 条, 验证集: {len(val_texts)} 条, 测试集: {len(test_texts)} 条")


# ================================================================
# 5. 加载已保存的数据集
# ================================================================

def load_dataset(data_dir="./processed_data"):
    """加载已保存的数据集（已分词）"""
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    val_df = pd.read_csv(f"{data_dir}/val.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    train_texts, train_labels = train_df['text'].tolist(), train_df['label'].tolist()
    val_texts, val_labels = val_df['text'].tolist(), val_df['label'].tolist()
    test_texts, test_labels = test_df['text'].tolist(), test_df['label'].tolist()

    print(f"✅ 加载数据集: 训练{len(train_texts)}, 验证{len(val_texts)}, 测试{len(test_texts)}")
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


# ================================================================
# 6. 主函数
# ================================================================

def main():
    """主函数：处理数据集"""
    print("=" * 50)
    print(f"处理数据集 (中文分词模式: {config_obj.zh_model})")
    print("=" * 50)

    # zh_texts, zh_labels = [], []
    # en_texts, en_labels = [], []
    #
    # if config_obj.chinese_data_path:
    #     zh_texts, zh_labels = load_chinese_data(config_obj.chinese_data_path)
    #     zh_texts, zh_labels = clean_texts(zh_texts, zh_labels)
    #     print(f"中文数据: {len(zh_texts)} 条")
    #
    # if config_obj.english_data_path:
    #     en_texts, en_labels = load_english_data(config_obj.english_data_path)
    #     en_texts, en_labels = clean_texts(en_texts, en_labels)
    #     print(f"英文数据: {len(en_texts)} 条")
    #
    # all_texts, all_labels = zh_texts + en_texts, zh_labels + en_labels
    # print(f"总数据: {len(all_texts)} 条")
    #
    # if zh_texts and en_texts:
    #     print("\n[中英文分别划分]")
    #     train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_mixed_data(
    #         zh_texts, zh_labels, en_texts, en_labels,
    #         config_obj.train_data_percent, config_obj.test_data_percent, config_obj.test_data_percent
    #     )
    # else:
    #     print("\n[整体划分]")
    #     train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_data(
    #         all_texts, all_labels,
    #         config_obj.train_data_percent, config_obj.test_data_percent, config_obj.test_data_percent
    #     )

    zh_texts, zh_labels = load_chinese_data(config_obj.data_zh_path)
    zh_texts, zh_labels = clean_texts(zh_texts, zh_labels)
    train_zh_texts, train_zh_labels, val_zh_texts, val_zh_labels, test_zh_texts, test_zh_labels = split_data(
                zh_texts, zh_labels,
                config_obj.train_data_percent, config_obj.test_data_percent, config_obj.test_data_percent
            )
    train_en_texts, train_en_labels = load_data(config_obj.train_en_path)
    val_en_texts, val_en_labels = load_data(config_obj.val_en_path)
    test_en_texts, test_en_labels = load_data(config_obj.test_en_path)

    print("\n[分词处理]")
    train_en_texts = tokenize_data(train_en_texts, config_obj.zh_model)
    val_en_texts = tokenize_data(val_en_texts, config_obj.zh_model)
    test_en_texts = tokenize_data(test_en_texts, config_obj.zh_model)

    train_texts = train_zh_texts + train_en_texts
    train_labels = train_zh_labels + train_en_labels
    val_texts = val_zh_texts + val_en_texts
    val_labels = val_zh_labels + val_en_labels
    test_texts = test_zh_texts + test_en_texts
    test_labels = test_zh_labels + test_en_labels

    print("\n[保存数据]")
    save_dataset(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, config_obj.data_process_path)
    print("=" * 50)


if __name__ == "__main__":
    main()
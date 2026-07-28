"""
数据预处理模块
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from config import *


def load_data(data_path=None):
    """
    加载数据，优先使用清洗后的数据

    Args:
        data_path: 数据文件路径，如果为None则自动查找

    Returns:
        texts: 文本列表
        labels: 标签列表
    """
    # 如果未指定路径，优先查找clean_train.csv
    if data_path is None:
        clean_path = os.path.join(DATA_DIR, CLEAN_DATA_FILE)
        raw_path = os.path.join(DATA_DIR, DATA_FILE)

        if os.path.exists(clean_path):
            data_path = clean_path
            text_col = CLEAN_TEXT_COLUMN
            print(f"使用清洗后数据: {CLEAN_DATA_FILE}")
        elif os.path.exists(raw_path):
            data_path = raw_path
            text_col = TEXT_COLUMN
            print(f"使用原始数据: {DATA_FILE}")
        else:
            raise FileNotFoundError(
                f"未找到数据文件，请确保 {DATA_DIR} 目录下存在 {DATA_FILE} 或 {CLEAN_DATA_FILE}"
            )
    else:
        # 根据文件名列名
        if 'clean' in data_path.lower():
            text_col = CLEAN_TEXT_COLUMN
        else:
            text_col = TEXT_COLUMN

    df = pd.read_csv(data_path)

    # 检查必要的列是否存在
    if text_col not in df.columns:
        # 如果指定的列不存在，尝试其他可能列名
        possible_cols = [TEXT_COLUMN, CLEAN_TEXT_COLUMN, 'text', 'content', 'review']
        for col in possible_cols:
            if col in df.columns:
                text_col = col
                break
        else:
            raise ValueError(f"未找到文本列，可用列: {df.columns.tolist()}")

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"未找到标签列 '{LABEL_COLUMN}'，可用列: {df.columns.tolist()}")

    texts = df[text_col].astype(str).tolist()
    labels = df[LABEL_COLUMN].values

    print(f"数据加载完成: {len(texts)} 条样本")
    print(f"标签分布: 正样本={sum(labels)}, 负样本={len(labels)-sum(labels)}")

    return texts, labels


def split_data(texts, labels):
    """
    划分数据集（8:1:1）
    先划分出测试集10%，再从剩余中划分验证集10%（即剩余数据的1/9）

    Args:
        texts: 文本列表
        labels: 标签列表

    Returns:
        train/val/test的数据和标签
    """
    # 先划分出训练集(80%)和临时集(20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    # 再从临时集中划分验证集(10%)和测试集(10%)，即临时集的一半
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    print(f"数据集划分完成:")
    print(f"  训练集: {len(X_train)} 条 ({len(X_train)/len(texts)*100:.1f}%)")
    print(f"  验证集: {len(X_val)} 条 ({len(X_val)/len(texts)*100:.1f}%)")
    print(f"  测试集: {len(X_test)} 条 ({len(X_test)/len(texts)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def vectorize_text(X_train, X_val, X_test):
    """
    TF-IDF向量化
    仅在训练集上拟合，然后转换所有数据

    Args:
        X_train: 训练文本
        X_val: 验证文本
        X_test: 测试文本

    Returns:
        向量化后的数据和vectorizer对象
    """
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE
    )

    # 仅在训练集上拟合
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    print(f"TF-IDF向量化完成:")
    print(f"  特征维度: {X_train_vec.shape[1]}")
    print(f"  ngram范围: {TFIDF_NGRAM_RANGE}")

    return X_train_vec, X_val_vec, X_test_vec, vectorizer


def prepare_data(data_path=None):
    """
    完整的数据准备流程

    Args:
        data_path: 数据文件路径，如果为None则使用默认路径

    Returns:
        包含所有处理数据的字典
    """
    # 加载数据
    texts, labels = load_data(data_path)

    # 划分数据
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(texts, labels)

    # 向量化
    X_train_vec, X_val_vec, X_test_vec, vectorizer = vectorize_text(
        X_train, X_val, X_test
    )

    # 保存vectorizer供后续使用
    vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"向量化器已保存至: {vectorizer_path}")

    return {
        'X_train': X_train_vec,
        'X_val': X_val_vec,
        'X_test': X_test_vec,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'vectorizer': vectorizer,
        'raw_texts': {
            'train': X_train,
            'val': X_val,
            'test': X_test
        }
    }


if __name__ == '__main__':
    # 测试数据预处理
    data = prepare_data()
    print("\n数据预处理完成，可以进行模型训练")

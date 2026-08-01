"""
词表构建模块
功能：加载已分词的数据集、构建/更新词表、保存词表
"""
import os
import json
from collections import Counter
from bilstm_config import Config
from bilstm_data_util import SPECIAL_TOKENS

config_obj = Config()


# ================================================================
# 1. 加载已分词的数据集
# ================================================================

def load_tokenized_data(data_dir="./processed_data"):
    """加载已分词的数据集"""
    import pandas as pd

    train_df = pd.read_csv(f"{data_dir}/train.csv")
    val_df = pd.read_csv(f"{data_dir}/val.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    train_texts = train_df['text'].tolist()
    val_texts = val_df['text'].tolist()
    test_texts = test_df['text'].tolist()

    print(f"✅ 加载已分词数据: 训练{len(train_texts)}, 验证{len(val_texts)}, 测试{len(test_texts)}")
    return train_texts, val_texts, test_texts


# ================================================================
# 2. 词表管理
# ================================================================

def build_vocab(texts, max_vocab=50000, min_freq = config_obj.min_freq):
    """从文本列表构建词表"""
    vocab = SPECIAL_TOKENS.copy()
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    for word, freq in counter.most_common(max_vocab - len(SPECIAL_TOKENS)):
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def save_vocab(vocab, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def update_vocab(vocab, texts, path, max_vocab=50000, min_freq= config_obj.min_freq):
    """更新词表，新词追加到后面"""
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    new_words = []
    for word, freq in counter.most_common():
        if freq >= min_freq and word not in vocab and word not in SPECIAL_TOKENS:
            new_words.append(word)

    if not new_words:
        return vocab, 0

    old_size = len(vocab)
    added = 0
    for word in new_words:
        if len(vocab) >= max_vocab:
            break
        vocab[word] = len(vocab)
        added += 1

    save_vocab(vocab, path)
    print(f"✅ 添加 {added} 个新词，词表从 {old_size} 扩展到 {len(vocab)}")
    return vocab, added


def get_vocab(texts, path, max_vocab=50000, is_create=False):
    # is_create 表示是否构建 True重新构建  False加载并更新
    if is_create:
        print("📝 构建新词表...")
        vocab = build_vocab(texts, max_vocab, config_obj.min_freq)
        save_vocab(vocab, path)
        return vocab
    else:
        """获取词表：存在则加载并更新，不存在则构建"""
        if not os.path.exists(path):
            print("📝 构建新词表...")
            vocab = build_vocab(texts, max_vocab,config_obj.min_freq)
            save_vocab(vocab, path)
            return vocab

        print("📂 加载词表...")
        vocab = load_vocab(path)
        print(f"   当前词表大小: {len(vocab)}")

        print("🔄 检测新词...")
        vocab, added = update_vocab(vocab, texts, path, max_vocab, config_obj.min_freq)
        if added == 0:
            print("   无新词，词表无需更新")

        return vocab


# ================================================================
# 3. 主函数
# ================================================================

def main():
    """主函数：构建词表"""
    print("=" * 50)
    print("构建词表")
    print("=" * 50)

    # 1. 加载已分词的数据
    train_texts, val_texts, test_texts = load_tokenized_data(config_obj.data_process_path)

    # 2. 构建词表（只用训练集）
    vocab = get_vocab(
        train_texts,
        config_obj.save_vocab_path + "/vocab.json",
        max_vocab=config_obj.vocab_size, is_create=True
    )

    print(f"\n✅ 词表构建完成，大小: {len(vocab)}")
    print(f"   特殊标记: {list(SPECIAL_TOKENS.keys())}")

    # 显示前10个词
    print("\n词表示例（前10个）:")
    for i, (word, idx) in enumerate(list(vocab.items())[:10]):
        print(f"   {idx}: {word}")

    return vocab


if __name__ == "__main__":
    main()
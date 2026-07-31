"""
批次数据构建模块
功能：加载已分词的数据集和词表、生成词序列、创建DataLoader批次处理（供训练使用）
"""
import torch
from torch.utils.data import Dataset, DataLoader
from bilstm_config import Config


config_obj = Config()


# ================================================================
# 1. 加载已分词的数据集和词表
# ================================================================

def load_data_and_vocab(data_dir="./processed_data", vocab_path=config_obj.save_vocab_path + "/vocab.json"):
    """加载已分词的数据集和词表"""
    import pandas as pd
    import json

    train_df = pd.read_csv(f"{data_dir}/train.csv")
    val_df = pd.read_csv(f"{data_dir}/val.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    train_texts, train_labels = train_df['text'].tolist(), train_df['label'].tolist()
    val_texts, val_labels = val_df['text'].tolist(), val_df['label'].tolist()
    test_texts, test_labels = test_df['text'].tolist(), test_df['label'].tolist()

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    print(f"✅ 加载完成:")
    print(f"   训练集: {len(train_texts)} 条")
    print(f"   验证集: {len(val_texts)} 条")
    print(f"   测试集: {len(test_texts)} 条")
    print(f"   词表大小: {len(vocab)}")

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, vocab


# ================================================================
# 2. 文本转词序列
# ================================================================

def text_to_ids(text, vocab, max_len=30):
    """将已分词的文本转换为Token ID序列"""
    tokens = ['[CLS]'] + text.split() + ['[SEP]']
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    seq_len = len(ids)
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [vocab['<PAD>']] * (max_len - len(ids))
    return ids, seq_len


# ================================================================
# 3. 数据集类
# ================================================================

class SentimentDataset(Dataset):
    """情感分析数据集类，将文本转换为词序列"""
    def __init__(self, texts, labels, vocab, max_len=30):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        ids, seq_len = text_to_ids(text, self.vocab, self.max_len)
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'seq_len': seq_len
        }


# ================================================================
# 4. 创建DataLoader（供训练使用）
# ================================================================

def create_dataloader(texts, labels, vocab, batch_size=16, shuffle=True, max_len=30, num_workers=0):
    """创建DataLoader，生成批次数据"""
    dataset = SentimentDataset(texts, labels, vocab, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=False)


# ================================================================
# 5. 构建批次数据（主函数）
# ================================================================

def build_batch_data(data_dir=config_obj.data_process_path,
                     vocab_path=config_obj.save_vocab_path + "/vocab.json",
                     max_vocab=config_obj.vocab_size,
                     max_len=config_obj.max_len,
                     batch_size=config_obj.batch_size):
    """
    构建批次数据（供训练使用）

    Returns:
        train_loader, val_loader, test_loader, vocab
    """
    print("=" * 50)
    print("生成词序列和批次数据")
    print("=" * 50)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, vocab = load_data_and_vocab(
        data_dir, vocab_path
    )

    # 2. 创建DataLoader
    print("\n[2] 创建DataLoader...")
    train_loader = create_dataloader(train_texts, train_labels, vocab, batch_size, True, max_len)
    val_loader = create_dataloader(val_texts, val_labels, vocab, batch_size, False, max_len)
    test_loader = create_dataloader(test_texts, test_labels, vocab, batch_size, False, max_len)

    print(f"\n✅ 批次数据创建完成:")
    print(f"   训练批次: {len(train_loader)}")
    print(f"   验证批次: {len(val_loader)}")
    print(f"   测试批次: {len(test_loader)}")
    print("=" * 50)

    return train_loader, val_loader, test_loader, vocab


# ================================================================
# 6. 主函数
# ================================================================

def main():
    """主函数：生成批次数据"""
    train_loader, val_loader, test_loader, vocab = build_batch_data()

    # 查看批次示例
    print("\n批次示例:")
    for batch in train_loader:
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  labels: {batch['label']}")
        print(f"  seq_len: {batch['seq_len']}")
        break


if __name__ == "__main__":
    main()
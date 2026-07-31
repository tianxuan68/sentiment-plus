"""
目的：保证代码与数据分隔，方便管理数据文件路径和参数

步骤
1. 导包
2. 定义配置类
3. 测试
"""

# 1. 导包
import torch

# 2. 定义配置类
class Config:
    def __init__(self):
        # 数据集路径（中英文）
        self.chinese_data_path = 'data/train.csv'
        self.english_data_path = 'data/data.csv'

        # 训练集、测试集、验证集比例为 8:1:1
        self.train_data_percent = 0.8
        self.test_data_percent = 0.1
        self.dev_data_percent = 0.1

        # 保存处理后数据路径
        self.data_process_path = './data_process'

        # 保存模型路径
        self.save_model_path = '../../../aspect_sentiment/bilstm'

        # 保存词表路径
        self.save_vocab_path = '../../../../artifacts/sentiment/bilstm'

        # 指标保存路径
        self.metric_data_path = '../../../../artifacts/metrics/bilstm'

        # 路演路径
        self.roadshow_data_path = '../../../../pitch_assets/chen_jiangping'

        # 保存最优参数路径
        self.save_best_param_path = './best_param'

        # ========== 数据 ==========
        self.max_len = 30  # 最大序列长度
        self.batch_size = 16  # 批次大小
        self.num_workers = 0  # Windows设为0
        self.random_seed = 68  # 随机种子
        self.min_freq = 2  # 最低词频

        # ========== 模型 ==========
        self.vocab_size = 10000  # 词表大小
        self.embed_dim = 32  # 词向量维度
        self.hidden_dim = 32  # LSTM隐藏层
        self.num_classes = 2  # 正向/负向
        self.num_layers = 2  # LSTM层数
        self.dropout = 0.5  # Dropout比例
        self.num_heads = 2  # 多头数量
        self.zh_model = 'char' # 中文分词方式
        self.weight_decay = 0.1 # L2正则化
        self.lr_factor = 0.5  # 学习率衰减因子
        self.lr_patience = 3  # 等待轮次
        self.ecope = 10 # 轮次

        # ========== 训练 ==========
        self.num_epochs = 30  # 训练轮数
        self.learning_rate = 1e-3  # 学习率
        self.patience = 3  # 早停耐心值

        # ========== 设备 ==========
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. 测试
if __name__ == '__main__':
    print("=" * 50)
    print("GPU PyTorch 检测")
    print("=" * 50)

    # 1. 检查 PyTorch 版本
    print(f"PyTorch 版本: {torch.__version__}")

    # 2. 检查 CUDA 是否可用（最关键！）
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")

    # 3. 如果可用，显示 GPU 信息
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.current_device()}")
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
    else:
        print("❌ CUDA 不可用，GPU 版 PyTorch 未正确安装")

    # 实例化配置类
    config_obj = Config()
    assert config_obj.save_model_path == '../../aspect_sentiment/bilstm'
    print("测试成功")
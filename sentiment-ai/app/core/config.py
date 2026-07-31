"""
应用级路径配置
"""

# 1.导包
from pathlib import Path


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(Path(__file__).resolve().parents[2]).replace('\\', '/') + '/'
        self.bert_model_dir = self.root_path + 'artifacts/bert/best'


config = Config()

# 兼容旧引用
AI_ROOT = Path(config.root_path)
BERT_MODEL_DIR = Path(config.bert_model_dir)


if __name__ == '__main__':
    print(f'AI 根目录：{config.root_path}')
    print(f'BERT 权重：{config.bert_model_dir}')

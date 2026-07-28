"""
Baseline模型配置文件
"""
import os

# 数据配置
# 优先使用刘攀清洗后的数据，如果没有则使用原始数据
DATA_FILE = 'train.csv'
CLEAN_DATA_FILE = 'clean_train.csv'
TEXT_COLUMN = 'sentence'
CLEAN_TEXT_COLUMN = 'text_clean'
LABEL_COLUMN = 'label'

# 划分配置
RANDOM_SEED = 68
TEST_SIZE = 0.1
VAL_SIZE = 0.1  # 从训练集中划分出验证集

# TF-IDF配置
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)

# 模型超参数（网格搜索用）
MODEL_PARAMS = {
    'knn': {
        'n_neighbors': [3, 5, 7, 9],
        'metric': ['euclidean', 'cosine']
    },
    'decision_tree': {
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [20, 30, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
}

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

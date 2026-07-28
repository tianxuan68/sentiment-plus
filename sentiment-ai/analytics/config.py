"""
满意度统计配置文件
"""
import os

# ============ 数据配置 ============
# 中文情感数据（train.csv）：sentence, label(0负/1正), dataset
CN_DATA_FILE = 'train.csv'
CN_TEXT_COLUMN = 'sentence'
CN_LABEL_COLUMN = 'label'

# 英文评分数据（data.csv）：reviewText, overall(0~4), asin
EN_DATA_FILE = 'data.csv'
EN_TEXT_COLUMN = 'reviewText'
EN_SCORE_COLUMN = 'overall'
EN_PRODUCT_COLUMN = 'asin'

# ============ 路径配置 ============
# analytics 目录本身
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据目录位于 sentiment-ai/data/raw
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')
# 输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')
CHART_DIR = os.path.join(OUTPUT_DIR, 'charts')

# 确保输出目录存在
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

# ============ 可视化配置 ============
# 中文字体
FONT_SANS_SERIF = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
# 图表尺寸
FIG_SIZE_PIE = (8, 8)
FIG_SIZE_BAR = (10, 6)
FIG_SIZE_KPI = (14, 4)
# 图片 DPI
DPI = 300
# 配色方案
COLORS_PIE = ['#ED7D31', '#4472C4']          # 负向(橙)、正向(蓝)
COLORS_BAR = '#4472C4'                        # 评分柱状图
COLORS_KPI = ['#4472C4', '#ED7D31', '#A5A5A5', '#70AD47']  # KPI 卡配色

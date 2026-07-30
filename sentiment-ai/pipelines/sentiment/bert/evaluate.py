"""在统一 test_split 上评估已导出的 BERT 权重。"""
# 该模块用于评估已训练好的BERT情感分类模型在测试集上的性能
# 主要功能包括：加载训练好的模型、在测试集上运行推理、计算评估指标、保存结果
# 评估指标包括准确率(Accuracy)、F1分数、详细的分类报告等
# 还会检查模型是否达到预设的验收标准（Acc>=0.92, F1>=0.90）

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer

# 从同包导入数据集类和评估函数
from dataset import SentimentDataset
from train import evaluate  # 复用训练模块中的评估函数

# sentiment-ai 项目根目录
# 通过当前文件路径向上查找3级目录获取项目根目录
AI_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: Path) -> dict:
    """加载YAML配置文件

    从指定路径读取YAML格式的配置文件，返回解析后的字典。
    配置文件包含模型路径、数据路径、评估参数等信息。

    Args:
        path: 配置文件的路径

    Returns:
        dict: 解析后的配置字典，包含以下典型键值：
            - processed_dir: 处理后的数据目录
            - output_dir: 输出目录
            - max_length: 文本最大长度
            - batch_size: 批次大小
            - model_name: 模型名称或路径
    """
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """主函数：执行BERT模型在测试集上的完整评估流程

    评估流程包括：
    1. 解析命令行参数（配置文件路径、模型目录）
    2. 加载配置文件
    3. 验证模型权重和测试数据文件是否存在
    4. 加载训练好的BERT模型和分词器
    5. 创建测试集数据加载器
    6. 在测试集上运行模型推理并计算评估指标
    7. 输出评估结果（准确率、F1分数、分类报告）
    8. 保存评估指标到JSON文件
    9. 检查是否达到验收标准并输出相应提示

    命令行参数：
        --config: YAML配置文件路径，默认为 configs/bert_sentiment.yaml
        --model_dir: 模型权重目录，默认为 artifacts/bert/best
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="BERT 测试集评估")
    parser.add_argument(
        "--config",
        type=Path,
        default=AI_ROOT / "configs" / "bert_sentiment.yaml",
        help="YAML配置文件路径",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help="权重目录，默认 artifacts/bert/best",
    )
    args = parser.parse_args()

    # 加载配置文件
    cfg = load_config(args.config)

    # 确定模型目录和测试数据路径
    # 如果命令行未指定模型目录，则使用配置文件中的默认值
    model_dir = args.model_dir or (AI_ROOT / cfg["output_dir"] / "best")
    test_csv = AI_ROOT / cfg["processed_dir"] / "test_split.csv"

    # 验证必要的文件是否存在
    if not model_dir.exists():
        raise FileNotFoundError(f"未找到权重: {model_dir}，请先训练")
    if not test_csv.exists():
        raise FileNotFoundError(f"未找到测试集: {test_csv}，请先划分")

    # 设置计算设备（优先使用GPU加速推理）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 从训练好的模型目录加载分词器和模型
    # 这些文件在训练时由train.py保存到best目录
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)

    # 创建测试集数据加载器
    # shuffle=False确保评估时数据顺序一致，便于结果复现
    loader = DataLoader(
        SentimentDataset(
            test_csv,
            tokenizer,
            max_length=int(cfg["max_length"])  # 文本截断/填充长度
        ),
        batch_size=int(cfg["batch_size"]),  # 每批次处理的样本数
        shuffle=False,  # 测试集不需要打乱
    )

    # 调用train模块中的evaluate函数进行模型评估
    # 该函数返回包含accuracy、f1、loss和详细分类报告的字典
    metrics = evaluate(model, loader, device)

    # 输出评估结果
    print(f"[evaluate] Acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f}")
    print(metrics["report"])  # 输出sklearn的详细分类报告（包含每个类别的精确率、召回率等）

    # 保存评估指标到JSON文件
    out = AI_ROOT / cfg["output_dir"] / "test_metrics.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": metrics["accuracy"],  # 准确率
                "f1": metrics["f1"],  # 加权F1分数
            },
            f,
            ensure_ascii=False,
            indent=2,  # 格式化输出便于阅读
        )
    print(f"saved -> {out}")

    # 检查是否达到验收标准
    # 验收目标：准确率>=0.92，F1分数>=0.90
    if metrics["accuracy"] < 0.92 or metrics["f1"] < 0.90:
        print("[warn] 未达验收: Acc>=0.92 且 F1>=0.90")
        # 可以在这里添加更多处理逻辑，如发送通知、记录日志等
    else:
        print("[ok] 达到验收目标")
        # 模型性能达标，可以进行后续的部署或应用


if __name__ == "__main__":
    # 脚本入口点
    # 当直接运行此脚本时，执行main函数
    # 可通过命令行参数自定义配置和模型路径
    main()

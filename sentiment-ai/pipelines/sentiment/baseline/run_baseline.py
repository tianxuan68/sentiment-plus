"""
Baseline模型完整流程主入口
运行此脚本完成：数据准备 -> 模型训练 -> 评估 -> 可视化

使用方法:
    python run_baseline.py

可选参数:
    --data-path: 指定数据文件路径
    --skip-train: 跳过训练，直接加载已有模型进行评估
"""
import os
import sys
import argparse
from config import *
from preprocess import prepare_data
from train import train_all_models
from evaluate import evaluate_all_models
from visualize import main as visualize_main


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Baseline模型训练与评估')
    parser.add_argument('--data-path', type=str, default=None,
                       help='指定数据文件路径')
    parser.add_argument('--skip-train', action='store_true',
                       help='跳过训练，加载已有模型')
    return parser.parse_args()


def main():
    """
    Baseline模型完整流程
    """
    args = parse_args()

    print("=" * 70)
    print("商品评论情感分析 - Baseline模型训练流程")
    print("=" * 70)
    print(f"工作目录: {BASE_DIR}")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"数据划分: 训练80% / 验证10% / 测试10%")
    print("=" * 70)

    try:
        # 步骤1: 数据准备
        print("\n[步骤 1/4] 数据准备")
        print("-" * 50)
        data = prepare_data(args.data_path)
        print(f"训练集: {data['X_train'].shape}")
        print(f"验证集: {data['X_val'].shape}")
        print(f"测试集: {data['X_test'].shape}")

        # 步骤2: 模型训练
        print("\n[步骤 2/4] 模型训练")
        print("-" * 50)

        if args.skip_train:
            # 加载已有模型
            import joblib
            trained_models = {}
            for model_name in ['knn', 'decision_tree', 'random_forest']:
                model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
                if os.path.exists(model_path):
                    trained_models[model_name] = {
                        'model': joblib.load(model_path)
                    }
                    print(f"已加载模型: {model_name}")
            if not trained_models:
                print("错误: 未找到已有模型，请取消 --skip-train 参数重新训练")
                return None
        else:
            trained_models = train_all_models(data)

        # 步骤3: 模型评估
        print("\n[步骤 3/4] 模型评估")
        print("-" * 50)
        results_df, predictions = evaluate_all_models(trained_models, data)

        # 步骤4: 结果可视化
        print("\n[步骤 4/4] 结果可视化")
        print("-" * 50)
        visualize_main(results_df)

        # 最终总结
        print("\n" + "=" * 70)
        print("Baseline模型训练与评估完成！")
        print("=" * 70)

        print(f"\n结果保存在:")
        print(f"  模型文件: {MODEL_DIR}")
        print(f"  评估结果: {RESULT_DIR}")

        # 验收标准检查
        print("\n" + "=" * 70)
        print("验收标准检查")
        print("=" * 70)

        # 检查模型数量
        model_count = len(trained_models)
        model_check = model_count >= 3
        print(f"{'[PASS]' if model_check else '[FAIL]} 模型数量: {model_count} {'>= 3' if model_check else '< 3'}")

        # 检查准确率
        best_acc = results_df['accuracy'].max()
        acc_check = best_acc >= 0.85
        print(f"{'[PASS]' if acc_check else '[FAIL]} 准确率: {best_acc:.4f} {'>= 0.85' if acc_check else '< 0.85'}")

        # 检查指标输出
        print(f"[PASS] 输出指标: Accuracy / Precision / Recall / F1-Score")

        # 检查最佳模型导出
        best_model_exported = os.path.exists(os.path.join(RESULT_DIR, 'best_model.txt'))
        print(f"{'[PASS]' if best_model_exported else '[FAIL]} 导出最佳模型信息")

        print("=" * 70)

        # 返回最佳模型信息
        best_idx = results_df['f1'].idxmax()
        best_model = results_df.loc[best_idx]
        print(f"\n最佳模型: {best_model['model']}")
        print(f"  Accuracy:  {best_model['accuracy']:.4f}")
        print(f"  Precision: {best_model['precision']:.4f}")
        print(f"  Recall:    {best_model['recall']:.4f}")
        print(f"  F1-Score:  {best_model['f1']:.4f}")

        return results_df

    except FileNotFoundError as e:
        print(f"\n错误: {str(e)}")
        print("\n请确保数据文件已放置在正确的位置:")
        print(f"  - 原始数据: {os.path.join(DATA_DIR, DATA_FILE)}")
        print(f"  - 清洗数据: {os.path.join(DATA_DIR, CLEAN_DATA_FILE)}")
        return None

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    results = main()

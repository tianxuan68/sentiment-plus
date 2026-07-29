"""
模型评估模块
"""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import pandas as pd
import numpy as np
import os
from config import *


def evaluate_model(model, X_test, y_test, model_name):
    """
    评估单个模型

    Args:
        model: 训练好的模型
        X_test: 测试数据
        y_test: 测试标签
        model_name: 模型名称

    Returns:
        评估指标字典和预测结果
    """
    # 预测
    y_pred = model.predict(X_test)

    # 计算指标
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }

    # 打印详细报告
    print(f"\n{'='*50}")
    print(f"{model_name} 模型评估结果")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['负向', '正向']))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"混淆矩阵:")
    print(f"         预测负  预测正")
    print(f"实际负   {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"实际正   {cm[1,0]:4d}    {cm[1,1]:4d}")

    return metrics, y_pred


def evaluate_all_models(trained_models, data):
    """
    评估所有模型

    Args:
        trained_models: 训练好的模型字典
        data: 数据字典

    Returns:
        评估结果DataFrame和预测结果字典
    """
    results = []
    predictions = {}

    print("\n" + "=" * 60)
    print("开始评估所有模型")
    print("=" * 60)

    for model_name, model_info in trained_models.items():
        metrics, y_pred = evaluate_model(
            model_info['model'],
            data['X_test'],
            data['y_test'],
            model_name
        )
        results.append(metrics)
        predictions[model_name] = y_pred

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    # 保存结果
    results_path = os.path.join(RESULT_DIR, 'metrics.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n评估结果已保存至: {results_path}")

    # 找出最佳模型（按F1分数）
    best_idx = results_df['f1'].idxmax()
    best_model = results_df.loc[best_idx]

    # 保存最佳模型信息
    best_model_path = os.path.join(RESULT_DIR, 'best_model.txt')
    with open(best_model_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("Baseline最佳模型\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"模型名称: {best_model['model']}\n")
        f.write(f"Accuracy:  {best_model['accuracy']:.4f}\n")
        f.write(f"Precision: {best_model['precision']:.4f}\n")
        f.write(f"Recall:    {best_model['recall']:.4f}\n")
        f.write(f"F1-Score:  {best_model['f1']:.4f}\n")

    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("\n" + "=" * 60)
    print(f"最佳模型: {best_model['model']}")
    print(f"测试集 F1-Score: {best_model['f1']:.4f}")
    print(f"测试集 Accuracy: {best_model['accuracy']:.4f}")
    print("=" * 60)

    return results_df, predictions


def main(trained_models=None, data=None):
    """
    主评估流程

    Args:
        trained_models: 训练好的模型字典，如果为None则自动加载
        data: 数据字典，如果为None则自动准备

    Returns:
        评估结果DataFrame和预测结果字典
    """
    print("=" * 60)
    print("Baseline模型评估")
    print("=" * 60)

    # 如果未提供数据，自动准备
    if data is None:
        from preprocess import prepare_data
        data = prepare_data()

    # 如果未提供模型，自动加载
    if trained_models is None:
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
            print("错误: 未找到训练好的模型，请先运行 train.py")
            return None, None

    results_df, predictions = evaluate_all_models(trained_models, data)

    return results_df, predictions


if __name__ == '__main__':
    results_df, predictions = main()

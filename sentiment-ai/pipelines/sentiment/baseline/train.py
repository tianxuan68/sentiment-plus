"""
模型训练模块
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os
from config import *
from preprocess import prepare_data


def get_models():
    """
    获取所有模型及其参数网格

    Returns:
        模型字典，key为模型名，value为(模型对象, 参数网格)
    """
    models = {
        'knn': (
            KNeighborsClassifier(),
            MODEL_PARAMS['knn']
        ),
        'decision_tree': (
            DecisionTreeClassifier(random_state=RANDOM_SEED),
            MODEL_PARAMS['decision_tree']
        ),
        'random_forest': (
            RandomForestClassifier(random_state=RANDOM_SEED),
            MODEL_PARAMS['random_forest']
        )
    }
    return models


def train_model(model, param_grid, X_train, y_train, X_val, y_val, model_name):
    """
    训练单个模型（使用网格搜索和验证集）

    Args:
        model: 模型对象
        param_grid: 参数网格
        X_train: 训练数据
        y_train: 训练标签
        X_val: 验证数据
        y_val: 验证标签
        model_name: 模型名称

    Returns:
        最佳模型和验证集分数
    """
    print(f"\n{'='*50}")
    print(f"训练 {model_name} 模型...")
    print(f"{'='*50}")

    # 网格搜索（3折交叉验证）
    grid_search = GridSearchCV(
        model, param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # 在验证集上评估
    val_score = grid_search.best_estimator_.score(X_val, y_val)

    print(f"\n{model_name} 训练完成:")
    print(f"  最佳参数: {grid_search.best_params_}")
    print(f"  验证集准确率: {val_score:.4f}")

    return grid_search.best_estimator_, val_score


def train_all_models(data):
    """
    训练所有模型

    Args:
        data: 包含训练和验证数据的字典

    Returns:
        训练好的模型字典
    """
    models_dict = get_models()
    trained_models = {}

    for model_name, (model, param_grid) in models_dict.items():
        best_model, val_score = train_model(
            model, param_grid,
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            model_name
        )

        trained_models[model_name] = {
            'model': best_model,
            'val_score': val_score
        }

        # 保存模型
        model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"  模型已保存至: {model_path}")

    return trained_models


def main():
    """
    主训练流程
    """
    print("=" * 60)
    print("Baseline模型训练")
    print("=" * 60)

    # 准备数据
    print("\n[1/2] 准备数据...")
    data = prepare_data()

    # 训练所有模型
    print("\n[2/2] 训练模型...")
    trained_models = train_all_models(data)

    print("\n" + "=" * 60)
    print("所有模型训练完成！")
    print("=" * 60)

    return trained_models, data


if __name__ == '__main__':
    trained_models, data = main()

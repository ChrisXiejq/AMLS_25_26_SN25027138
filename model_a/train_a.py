import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from .data_loader_a import load_numpy
from .plot_a import plot_capacity_performance, plot_conf_matrix, plot_train_budget, ensure_dir


def build_model(capacity=1.0, use_pca=True, pca_dim=50):
    """
    构建 Model A：SVM + (可选) PCA
    capacity = SVM 的 C
    """
    steps = []

    # 1. 标准化
    steps.append(("scaler", StandardScaler()))

    # 2. PCA（可选）
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_dim)))

    # 3. SVM
    svm = SVC(C=capacity, kernel="rbf", gamma="scale")
    steps.append(("svm", svm))

    return Pipeline(steps)


def evaluate_model(model, X_test, y_test):
    """返回 accuracy / precision / recall / f1"""
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def train_model_a(
    use_pca=True,
    pca_dim=50,
    capacity_list=[0.1, 1, 10],
    subset_ratio=1.0,     # 训练预算
):
    """
    整个 Model A 训练 Pipeline：
    - 加载数据
    - 训练多种 capacity (C)
    - 可调 PCA、训练预算
    """

    # 1. 加载 numpy 数据
    X_train, y_train = load_numpy("train")
    X_val, y_val = load_numpy("val")
    X_test, y_test = load_numpy("test")

    # 2. 使用训练预算（如 0.2 代表 20% 数据）
    if subset_ratio < 1.0:
        n = int(len(X_train) * subset_ratio)
        X_train = X_train[:n]
        y_train = y_train[:n]
        print(f"[Training Budget] Using subset: {n}/{len(X_train)} samples")

    results = {}

    for C in capacity_list:
        print(f"\n===== Training SVM (C={C}) =====")

        model = build_model(capacity=C, use_pca=use_pca, pca_dim=pca_dim)

        model.fit(X_train, y_train)

        # 评估
        res_val = evaluate_model(model, X_val, y_val)
        res_test = evaluate_model(model, X_test, y_test)

        results[C] = {
            "val": res_val,
            "test": res_test,
            "model": model,
        }

        print(f"[Val]  acc={res_val['accuracy']:.4f}, f1={res_val['f1']:.4f}")
        print(f"[Test] acc={res_test['accuracy']:.4f}, f1={res_test['f1']:.4f}")

        # --- 保存可视化 ---
    output_dir = "outputs/model_a"
    ensure_dir(output_dir)

    # ① capacity vs performance
    plot_capacity_performance(results, output_dir)

    # ② confusion matrix，用最好 C 的模型
    best_C = max(results.keys(), key=lambda c: results[c]["test"]["accuracy"])
    plot_conf_matrix(results, best_C, X_test, y_test, output_dir)

    return results


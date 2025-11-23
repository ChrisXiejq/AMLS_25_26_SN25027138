import os
import sys
from pathlib import Path
from model_a.data_loader_a import load_numpy
from model_b.data_loader_b import get_dataloaders

from model_a.train_a import train_model_a
from model_a.eval_a import report_results

from model_b.train_b import train_model_b


def get_breastmnist_root() -> str:
    """
    返回数据根目录，按照作业要求使用项目根目录下的 Datasets 文件夹。
    """

    this_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_root = os.path.join(this_dir, "Datasets")
    os.makedirs(datasets_root, exist_ok=True)
    return datasets_root

def check_model_a_data():
    print("=== Model A: numpy data ===")
    X_train, y_train = load_numpy("train")
    print("Train:", X_train.shape, y_train.shape)

def check_model_b_data():
    print("=== Model B: torch dataloaders ===")
    train_loader, val_loader, _ = get_dataloaders()
    xb, yb = next(iter(train_loader))
    print("Batch:", xb.shape, yb.shape)

def run_model_a_experiments():
    print("\n========= Running Model A Experiments (SVM + PCA) =========\n")

    results = train_model_a(
        use_pca=True,
        pca_dim=50,
        capacity_list=[0.1, 1, 10],
        subset_ratio=1.0,  # 可改成 0.2 测试训练预算
    )

    report_results(results)

def run_model_b_experiments():
    print("\n========= Running Model B (CNN) =========\n")
    results = train_model_b(
        batch_size=64,
        lr=1e-3,
        epochs=10,
        augment=True,
    )
    return results


if __name__ == "__main__":
    get_breastmnist_root()
    check_model_a_data()
    check_model_b_data()
    print("Data check finished.")
    run_model_a_experiments()
    run_model_b_experiments()


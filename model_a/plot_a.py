import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_capacity_performance(results, save_dir):
    """
        visualize: capacity vs performance
        args:
            results: dict from train_model_a
            save_dir: directory to save plots
    """
    ensure_dir(save_dir)

    Cs = list(results.keys())
    accs = [results[C]["test"]["accuracy"] for C in Cs]
    f1s = [results[C]["test"]["f1"] for C in Cs]

    plt.figure(figsize=(8, 5))
    plt.plot(Cs, accs, marker="o", label="Accuracy")
    plt.plot(Cs, f1s, marker="s", label="F1-score")
    plt.xscale("log")
    plt.xlabel("SVM Capacity C (log scale)")
    plt.ylabel("Score")
    plt.title("Model A: Performance vs Capacity")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, "capacity_performance.png"), dpi=300)
    plt.close()


def plot_conf_matrix(results, C_best, X_test, y_test, save_dir):
    """ visualize: confusion matrix """
    ensure_dir(save_dir)

    model = results[C_best]["model"]
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix (C={C_best})")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


def plot_train_budget(budget_results, save_dir):
    """visualize: training budget vs performance"""
    ensure_dir(save_dir)

    ratios = []
    accs = []

    for ratio, res in budget_results.items():
        ratios.append(ratio)
        accs.append(res["test"]["accuracy"])

    plt.figure(figsize=(8, 5))
    plt.plot(ratios, accs, marker="o")
    plt.xlabel("Training Data Ratio")
    plt.ylabel("Test Accuracy")
    plt.title("Model A: Performance vs Training Budget")
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, "training_budget.png"), dpi=300)
    plt.close()

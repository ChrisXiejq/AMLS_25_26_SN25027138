import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from .plot_b import plot_confusion_matrix


def evaluate_metrics(model, test_loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)

            outputs = model(xb)
            preds = outputs.argmax(1)

            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

def report_results(results_dict):
    """
    输入 train_model_b() 的返回结果
    格式化输出，用于写报告
    """
    print("\n================= MODEL B SUMMARY =================")
    for metric, value in results_dict.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print("====================================================\n")

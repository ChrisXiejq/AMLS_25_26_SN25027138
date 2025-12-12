import torch
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from .plot_b import plot_confusion_matrix

def evaluate_metrics(model, dataloader, device, return_preds=False):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            p = out.argmax(1)

            preds.extend(p.cpu().numpy())
            labels.extend(yb.cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    if return_preds:
        return metrics, preds, labels
    else:
        return metrics

import torch
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_metrics(model, dataloader, device, return_preds=False, return_probs=False):
    """
    Evaluate classification metrics for the given model and dataloader.
    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        device (torch.device): Device to run the evaluation on.
        return_preds (bool): Whether to return predictions and labels.
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1-score.
        (optional) np.ndarray: Predictions array if return_preds is True.
        (optional) np.ndarray: Labels array if return_preds is True.
    """
    model.eval()
    preds = []
    labels = []
    probs = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            p = out.argmax(1)
            prob = torch.softmax(out, dim=1)[:, 1]

            preds.extend(p.cpu().numpy())
            labels.extend(yb.cpu().numpy())
            probs.extend(prob.cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)
    probs = np.array(probs)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    roc_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }

    if return_preds and return_probs:
        return metrics, preds, labels, probs
    if return_preds:
        return metrics, preds, labels
    if return_probs:
        return metrics, probs
    return metrics

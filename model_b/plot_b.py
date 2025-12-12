# model_b/plot_b.py
import matplotlib.pyplot as plt
import os
import seaborn as sns

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_learning_curve(train_losses, val_losses, val_accuracies, suffix=""):
    ensure_dir("outputs/model_b")

    if suffix:
        suffix = f"_{suffix}"

    # Loss curve
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title(f"Loss Curve{suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"outputs/model_b/loss_curve{suffix}.png", dpi=300)
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.legend()
    plt.title(f"Accuracy Curve{suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"outputs/model_b/accuracy_curve{suffix}.png", dpi=300)
    plt.close()


def plot_confusion_matrix(cm, suffix=""):
    """
    Save confusion matrix with experiment suffix:
    e.g. confusion_small.png, confusion_large_augmented.png
    """
    ensure_dir("outputs/model_b")

    if suffix:
        suffix = f"_{suffix}"

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix{suffix}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"outputs/model_b/confusion_matrix{suffix}.png", dpi=300)
    plt.close()

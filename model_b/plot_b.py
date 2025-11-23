import matplotlib.pyplot as plt
import os
import seaborn as sns


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_learning_curve(train_losses, val_losses, val_acc):
    ensure_dir("outputs/model_b")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("CNN Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("outputs/model_b/learning_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(val_acc, label="Val Accuracy", marker="o")
    plt.title("Validation Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig("outputs/model_b/val_accuracy_curve.png", dpi=300)
    plt.close()


def plot_confusion_matrix(cm):
    ensure_dir("outputs/model_b")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("CNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("outputs/model_b/confusion_matrix.png", dpi=300)
    plt.close()

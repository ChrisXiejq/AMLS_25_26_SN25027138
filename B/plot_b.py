# B/plot_b.py
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_five_dimension_comparison(all_results, run_dir):
    """
    Create five comparison plots for different dimensions:
    Dimension 1: Augmentation impact (no aug vs with aug)
    Dimension 2: Model capacity impact (small, medium, large)
    Dimension 3: Training budget impact (different ratios)
    Dimension 4: Optimizer impact (adam, sgd, rmsprop)
    Dimension 5: Loss function impact (crossentropy vs focal)
    
    args:
        all_results: dict with dimension data
        run_dir: directory to save plots
    """
    ensure_dir(run_dir)
    
    # Plot 1: Augmentation Impact
    plot_augmentation_impact(all_results, run_dir)
    
    # Plot 2: Capacity Impact
    plot_capacity_impact(all_results, run_dir)
    
    # Plot 3: Budget Impact
    plot_budget_impact(all_results, run_dir)
    
    # Plot 4: Optimizer Impact
    plot_optimizer_impact(all_results, run_dir)
    
    # Plot 5: Loss Function Impact
    plot_loss_function_impact(all_results, run_dir)


def plot_four_dimension_comparison(all_results, run_dir):
    """
    Create four comparison plots for different dimensions:
    Dimension 1: Augmentation impact (no aug vs with aug)
    Dimension 2: Model capacity impact (small, medium, large)
    Dimension 3: Training budget impact (different ratios)
    Dimension 4: Optimizer impact (adam, sgd, rmsprop)
    
    args:
        all_results: dict with dimension data
        run_dir: directory to save plots
    """
    ensure_dir(run_dir)
    
    # Plot 1: Augmentation Impact
    plot_augmentation_impact(all_results, run_dir)
    
    # Plot 2: Capacity Impact
    plot_capacity_impact(all_results, run_dir)
    
    # Plot 3: Budget Impact
    plot_budget_impact(all_results, run_dir)
    
    # Plot 4: Optimizer Impact
    plot_optimizer_impact(all_results, run_dir)


def plot_augmentation_impact(all_results, run_dir):
    """
    Dimension 1: Compare No Aug vs With Aug
    Shows impact of data augmentation
    args:
        all_results: dict with dimension data
        run_dir: directory to save plots
    """
    no_aug = all_results["dim1_no_aug"]
    aug = all_results["dim1_aug"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x_pos = np.array([0, 1])
    labels = ["No Aug", "With Aug"]
    
    # Collect metrics
    accs = [no_aug["accuracy"], aug["accuracy"]]
    f1s = [no_aug["f1"], aug["f1"]]
    colors = ["#FF6B6B", "#4ECDC4"]
    
    # Accuracy plot
    bars1 = axes[0].bar(x_pos, accs, width=0.6, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, fontsize=12)
    axes[0].set_ylabel("Test Accuracy", fontsize=11)
    axes[0].set_title("Dimension 1: Augmentation Impact", fontsize=12, fontweight="bold")
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.03, f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    
    # F1-score plot
    bars2 = axes[1].bar(x_pos, f1s, width=0.6, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, fontsize=12)
    axes[1].set_ylabel("Test F1-score", fontsize=11)
    axes[1].set_title("Dimension 1: Augmentation Impact (F1-score)", fontsize=12, fontweight="bold")
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.03, f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    
    # Add improvement annotation
    improvement = (aug["accuracy"] - no_aug["accuracy"]) * 100
    axes[0].text(0.5, 0.95, f"Improvement: {improvement:+.2f}%", 
                ha="center", va="top", transform=axes[0].transAxes,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "dim1_augmentation_impact.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: dim1_augmentation_impact.png")


def plot_capacity_impact(all_results, run_dir):
    """
    Dimension 2: Compare different model capacities (small, medium, large)
    Shows how model size affects performance
    """
    capacity_data = all_results["dim2_capacity"]
    
    sizes = ["small", "medium", "large"]
    accs = [capacity_data[s]["accuracy"] for s in sizes]
    f1s = [capacity_data[s]["f1"] for s in sizes]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x_pos = np.arange(len(sizes))
    colors = ["#95E1D3", "#4ECDC4", "#2C7873"]
    
    # Accuracy plot
    bars1 = axes[0].bar(x_pos, accs, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([s.capitalize() for s in sizes], fontsize=11)
    axes[0].set_ylabel("Test Accuracy", fontsize=11)
    axes[0].set_title("Dimension 2: Model Capacity Impact", fontsize=12, fontweight="bold")
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.02, f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    
    # F1-score plot
    bars2 = axes[1].bar(x_pos, f1s, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([s.capitalize() for s in sizes], fontsize=11)
    axes[1].set_ylabel("Test F1-score", fontsize=11)
    axes[1].set_title("Dimension 2: Model Capacity Impact (F1-score)", fontsize=12, fontweight="bold")
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.02, f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "dim2_capacity_impact.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: dim2_capacity_impact.png")


def plot_budget_impact(all_results, run_dir):
    """
    Dimension 3: Compare different training budgets
    Shows how training data ratio affects performance
    """
    budget_data = all_results["dim3_budgets"]
    
    ratios = sorted(budget_data.keys())
    accs = [budget_data[r]["accuracy"] for r in ratios]
    f1s = [budget_data[r]["f1"] for r in ratios]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x_labels = [f"{r*100:.0f}%" for r in ratios]
    x_pos = np.arange(len(ratios))
    colors_gradient = plt.cm.Blues(np.linspace(0.4, 0.9, len(ratios)))
    
    # Accuracy plot
    bars1 = axes[0].bar(x_pos, accs, color=colors_gradient, alpha=0.8, edgecolor="black", linewidth=1.5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(x_labels, fontsize=11)
    axes[0].set_ylabel("Test Accuracy", fontsize=11)
    axes[0].set_xlabel("Training Data Ratio", fontsize=11)
    axes[0].set_title("Dimension 3: Training Budget Impact", fontsize=12, fontweight="bold")
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.02, f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    
    # F1-score plot
    bars2 = axes[1].bar(x_pos, f1s, color=colors_gradient, alpha=0.8, edgecolor="black", linewidth=1.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(x_labels, fontsize=11)
    axes[1].set_ylabel("Test F1-score", fontsize=11)
    axes[1].set_xlabel("Training Data Ratio", fontsize=11)
    axes[1].set_title("Dimension 3: Training Budget Impact (F1-score)", fontsize=12, fontweight="bold")
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.02, f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "dim3_budget_impact.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: dim3_budget_impact.png")


def plot_optimizer_impact(all_results, run_dir):
    """
    Dimension 4: Compare different optimizers (adam, sgd, rmsprop)
    Shows how optimizer choice affects performance
    """
    optimizer_data = all_results["dim4_optimizers"]
    
    optimizers = ["adam", "sgd", "rmsprop"]
    accs = [optimizer_data[opt]["accuracy"] for opt in optimizers]
    f1s = [optimizer_data[opt]["f1"] for opt in optimizers]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x_pos = np.arange(len(optimizers))
    colors = ["#FFA07A", "#FFD700", "#87CEEB"]
    
    # Accuracy plot
    bars1 = axes[0].bar(x_pos, accs, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([opt.upper() for opt in optimizers], fontsize=11)
    axes[0].set_ylabel("Test Accuracy", fontsize=11)
    axes[0].set_title("Dimension 4: Optimizer Impact", fontsize=12, fontweight="bold")
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.02, f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    
    # F1-score plot
    bars2 = axes[1].bar(x_pos, f1s, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([opt.upper() for opt in optimizers], fontsize=11)
    axes[1].set_ylabel("Test F1-score", fontsize=11)
    axes[1].set_title("Dimension 4: Optimizer Impact (F1-score)", fontsize=12, fontweight="bold")
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.02, f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "dim4_optimizer_impact.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: dim4_optimizer_impact.png")


def plot_loss_function_impact(all_results, run_dir):
    """
    Dimension 5: Compare different loss functions (crossentropy vs focal)
    Shows how loss function choice affects performance
    """
    loss_data = all_results["dim5_loss_functions"]
    
    loss_fns = ["crossentropy", "focal"]
    accs = [loss_data[lf]["accuracy"] for lf in loss_fns]
    f1s = [loss_data[lf]["f1"] for lf in loss_fns]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x_pos = np.arange(len(loss_fns))
    labels = ["CrossEntropy", "Focal Loss"]
    colors = ["#FF6B9D", "#C44569"]
    
    # Accuracy plot
    bars1 = axes[0].bar(x_pos, accs, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, fontsize=11)
    axes[0].set_ylabel("Test Accuracy", fontsize=11)
    axes[0].set_title("Dimension 5: Loss Function Impact", fontsize=12, fontweight="bold")
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.02, f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    
    # F1-score plot
    bars2 = axes[1].bar(x_pos, f1s, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, fontsize=11)
    axes[1].set_ylabel("Test F1-score", fontsize=11)
    axes[1].set_title("Dimension 5: Loss Function Impact (F1-score)", fontsize=12, fontweight="bold")
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.02, f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    
    # Add comparison note
    diff = (accs[1] - accs[0]) * 100
    note = f"Focal vs CE: {diff:+.2f}%"
    axes[0].text(0.5, 0.05, note, ha="center", va="bottom", transform=axes[0].transAxes,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
                fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "dim5_loss_function_impact.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: dim5_loss_function_impact.png")


def plot_learning_curve(train_losses, val_losses, val_accuracies, suffix="", log_dir=None):
    """
    Save learning curves with experiment suffix:
    e.g. loss_curve_small.png, accuracy_curve_large_augmented.png
    """
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
    plt.savefig(f"{log_dir}/loss_curve{suffix}.png", dpi=300)
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.legend()
    plt.title(f"Accuracy Curve{suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"{log_dir}/accuracy_curve{suffix}.png", dpi=300)
    plt.close()


def plot_confusion_matrix(cm, suffix="", log_dir=None):
    """
    Save confusion matrix with experiment suffix:
    e.g. confusion_small.png, confusion_large_augmented.png
    """

    if suffix:
        suffix = f"_{suffix}"

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix{suffix}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{log_dir}/confusion_matrix{suffix}.png", dpi=300)
    plt.close()


def plot_roc_curve(labels, probs, suffix="", log_dir=None):
    """Plot ROC curve using predicted probabilities for the positive class."""
    if suffix:
        suffix = f"_{suffix}"

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#4ECDC4", lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve{suffix}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    ensure_dir(log_dir)
    plt.savefig(f"{log_dir}/roc_curve{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()

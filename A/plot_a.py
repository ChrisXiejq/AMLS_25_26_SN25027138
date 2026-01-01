import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def ensure_dir(path):
    """Ensure directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def plot_four_dimension_comparison(all_results, run_dir, best_C):
    """
    Create four comparison plots for different dimensions:
    Dimension 1: Raw vs Processed (pre-processing impact)
    Dimension 2: Processed (No Aug) vs Processed (With Aug) - augmentation impact
    Dimension 3: Processed+Aug with different C values - capacity impact
    Dimension 4: Processed+Aug+Best_C with different budgets - budget impact
    
    args:
        all_results: dict with dimension data
        run_dir: directory to save plots
        best_C: the best capacity value found
    """
    ensure_dir(run_dir)
    
    # ===== Plot 1: Pre-processing Impact (Raw vs Processed) =====
    plot_preprocessing_impact(all_results, run_dir)
    
    # ===== Plot 2: Augmentation Impact (Processed: No Aug vs With Aug) =====
    plot_augmentation_impact(all_results, run_dir)
    
    # ===== Plot 3: Capacity Impact (Processed+Aug: different C values) =====
    plot_capacity_impact(all_results, run_dir, best_C)
    
    # ===== Plot 4: Training Budget Impact (Processed+Aug+Best_C: different budgets) =====
    plot_budget_impact(all_results, run_dir, best_C)


def plot_preprocessing_impact(all_results, run_dir):
    """
    Dimension 1: Compare RAW vs PROCESSED features
    Shows impact of HOG+PCA preprocessing
    """
    results_raw = all_results["dim1_raw"]
    results_processed = all_results["dim1_processed"]
    
    # Get best C and its performance for each
    best_C_raw = max(results_raw.keys(), key=lambda c: results_raw[c]["test"]["accuracy"])
    best_C_proc = max(results_processed.keys(), key=lambda c: results_processed[c]["test"]["accuracy"])
    
    acc_raw = results_raw[best_C_raw]["test"]["accuracy"]
    f1_raw = results_raw[best_C_raw]["test"]["f1"]
    acc_proc = results_processed[best_C_proc]["test"]["accuracy"]
    f1_proc = results_processed[best_C_proc]["test"]["f1"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x_pos = np.array([0, 1])
    labels = ["Raw", "Processed\n(HOG+PCA)"]
    accs = [acc_raw, acc_proc]
    f1s = [f1_raw, f1_proc]
    colors = ["#FF6B6B", "#4ECDC4"]
    
    # Accuracy plot
    bars1 = axes[0].bar(x_pos, accs, width=0.6, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, fontsize=11)
    axes[0].set_ylabel("Test Accuracy", fontsize=11)
    axes[0].set_title("Dimension 1: Pre-processing Impact\n(Best C for each)", fontsize=12, fontweight="bold")
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.03, f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    
    # F1-score plot
    bars2 = axes[1].bar(x_pos, f1s, width=0.6, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, fontsize=11)
    axes[1].set_ylabel("Test F1-score", fontsize=11)
    axes[1].set_title("Dimension 1: Pre-processing Impact (F1-score)\n(Best C for each)", fontsize=12, fontweight="bold")
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.03, f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "dim1_preprocessing_impact.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: dim1_preprocessing_impact.png")


def plot_augmentation_impact(all_results, run_dir):
    """
    Dimension 2: Compare Processed (No Aug) vs Processed (With Aug)
    Shows impact of data augmentation
    """
    results_no_aug = all_results["dim2_processed_no_aug"]
    results_aug = all_results["dim2_processed_aug"]
    
    # Get best C and performance for each
    best_C_no_aug = max(results_no_aug.keys(), key=lambda c: results_no_aug[c]["test"]["accuracy"])
    best_C_aug = max(results_aug.keys(), key=lambda c: results_aug[c]["test"]["accuracy"])
    
    acc_no_aug = results_no_aug[best_C_no_aug]["test"]["accuracy"]
    f1_no_aug = results_no_aug[best_C_no_aug]["test"]["f1"]
    acc_aug = results_aug[best_C_aug]["test"]["accuracy"]
    f1_aug = results_aug[best_C_aug]["test"]["f1"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x_pos = np.array([0, 1])
    labels = ["Processed\n(No Aug)", "Processed\n(With Aug)"]
    accs = [acc_no_aug, acc_aug]
    f1s = [f1_no_aug, f1_aug]
    colors = ["#95E1D3", "#4ECDC4"]
    
    # Accuracy plot
    bars1 = axes[0].bar(x_pos, accs, width=0.6, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, fontsize=11)
    axes[0].set_ylabel("Test Accuracy", fontsize=11)
    axes[0].set_title("Dimension 2: Augmentation Impact\n(Best C for each)", fontsize=12, fontweight="bold")
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.03, f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    
    # F1-score plot
    bars2 = axes[1].bar(x_pos, f1s, width=0.6, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, fontsize=11)
    axes[1].set_ylabel("Test F1-score", fontsize=11)
    axes[1].set_title("Dimension 2: Augmentation Impact (F1-score)\n(Best C for each)", fontsize=12, fontweight="bold")
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.03, f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    
    # Add improvement annotation
    improvement = (acc_aug - acc_no_aug) * 100
    axes[0].text(0.5, 0.95, f"Improvement: {improvement:+.2f}%", 
                ha="center", va="top", transform=axes[0].transAxes,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "dim2_augmentation_impact.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: dim2_augmentation_impact.png")


def plot_capacity_impact(all_results, run_dir, best_C):
    """
    Dimension 3: Compare different capacity (C) values using Processed+Aug
    Shows how different C values affect performance
    """
    results = all_results["dim3_capacity"]
    
    Cs = sorted(results.keys())
    accs = [results[C]["test"]["accuracy"] for C in Cs]
    f1s = [results[C]["test"]["f1"] for C in Cs]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(Cs, accs, marker="o", linewidth=2.5, markersize=10, 
                label="Accuracy", color="#4ECDC4")
    axes[0].axvline(x=best_C, color="red", linestyle="--", linewidth=2, label=f"Best C={best_C}")
    axes[0].set_xlabel("SVM Capacity C (log scale)", fontsize=11)
    axes[0].set_ylabel("Test Accuracy", fontsize=11)
    axes[0].set_title("Dimension 3: Model Capacity Impact\n(Processed + Augmented)", fontsize=12, fontweight="bold")
    axes[0].set_xscale("log")
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # F1-score plot
    axes[1].plot(Cs, f1s, marker="s", linewidth=2.5, markersize=10, 
                label="F1-score", color="#95E1D3")
    axes[1].axvline(x=best_C, color="red", linestyle="--", linewidth=2, label=f"Best C={best_C}")
    axes[1].set_xlabel("SVM Capacity C (log scale)", fontsize=11)
    axes[1].set_ylabel("Test F1-score", fontsize=11)
    axes[1].set_title("Dimension 3: Model Capacity Impact (F1-score)\n(Processed + Augmented)", fontsize=12, fontweight="bold")
    axes[1].set_xscale("log")
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "dim3_capacity_impact.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: dim3_capacity_impact.png")


def plot_budget_impact(all_results, run_dir, best_C):
    """
    Dimension 4: Compare different training budgets using Processed+Aug+Best_C
    Shows how training data ratio affects performance
    """
    budget_data = all_results["dim4_budgets"]
    
    ratios = sorted(budget_data.keys())
    accs = [budget_data[r]["test"]["accuracy"] for r in ratios]
    f1s = [budget_data[r]["test"]["f1"] for r in ratios]
    
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
    axes[0].set_title(f"Dimension 4: Training Budget Impact\n(Processed + Augmented + C={best_C})", fontsize=12, fontweight="bold")
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
    axes[1].set_title(f"Dimension 4: Training Budget Impact (F1-score)\n(Processed + Augmented + C={best_C})", fontsize=12, fontweight="bold")
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.02, f"{v:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "dim4_budget_impact.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: dim4_budget_impact.png")


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

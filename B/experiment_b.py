import os
from datetime import datetime
from .train_b import train_model_b
from .plot_b import plot_five_dimension_comparison

def run_model_b_experiments():
    """
    Run five-dimensional experiments for Model B:
    Dimension 1: Augmentation (no aug vs with aug)
    Dimension 2: Model capacity (small, medium, large)
    Dimension 3: Training budget (different data ratios)
    Dimension 4: Optimizer (adam, sgd, rmsprop)
        Dimension 5: Loss function (crossentropy vs focal)
    """

    # Create a unique RUN directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"B/logs/run_{timestamp}"
    print(f"\n========= MODEL B EXPERIMENTS: {run_dir} =========\n")

    all_results = {}

    # DIMENSION 1: AUGMENTATION COMPARISON
    print("\n[Dimension 1] Augmentation: No Aug vs With Aug")
    print("=" * 50)
    
    # No augmentation
    print("\n  > Training WITHOUT augmentation...")
    no_aug_dir = f"{run_dir}/dim1_no_aug"
    results_no_aug = train_model_b(
        augment=False,
        model_size="medium",
        subset_ratio=1.0,
        log_dir=no_aug_dir
    )
    report_results(results_no_aug, "No Augmentation")
    all_results["dim1_no_aug"] = results_no_aug

    # With augmentation
    print("\n  > Training WITH augmentation...")
    aug_dir = f"{run_dir}/dim1_aug"
    results_aug = train_model_b(
        augment=True,
        model_size="medium",
        subset_ratio=1.0,
        log_dir=aug_dir
    )
    report_results(results_aug, "With Augmentation")
    all_results["dim1_aug"] = results_aug

    # DIMENSION 2: CAPACITY COMPARISON
    print("\n[Dimension 2] Model Capacity: Small, Medium, Large")
    print("=" * 50)
    
    all_results["dim2_capacity"] = {}
    
    for size in ["small", "medium", "large"]:
        print(f"\n  > Training {size.upper()} model...")
        cap_dir = f"{run_dir}/dim2_{size}"
        result = train_model_b(
            augment=True,
            model_size=size,
            subset_ratio=1.0,
            log_dir=cap_dir
        )
        report_results(result, f"Capacity: {size}")
        all_results["dim2_capacity"][size] = result

    # DIMENSION 3: TRAINING BUDGET COMPARISON
    print("\n[Dimension 3] Training Budget: Different data ratios")
    print("=" * 50)
    
    budget_ratios = [0.1, 0.3, 0.5, 1.0]
    all_results["dim3_budgets"] = {}
    
    for ratio in budget_ratios:
        print(f"\n  > Training with {ratio*100:.0f}% training data...")
        budget_dir = f"{run_dir}/dim3_budget_{ratio}"
        result = train_model_b(
            augment=True,
            model_size="medium",
            subset_ratio=ratio,
            log_dir=budget_dir
        )
        report_results(result, f"Budget {ratio*100:.0f}%")
        all_results["dim3_budgets"][ratio] = result

    # DIMENSION 4: OPTIMIZER COMPARISON
    print("\n[Dimension 4] Optimizer: Adam, SGD, RMSprop")
    print("=" * 50)
    
    all_results["dim4_optimizers"] = {}
    
    for opt in ["adam", "sgd", "rmsprop"]:
        print(f"\n  > Training with {opt.upper()} optimizer...")
        opt_dir = f"{run_dir}/dim4_{opt}"
        result = train_model_b(
            augment=True,
            model_size="medium",
            subset_ratio=1.0,
            optimizer_name=opt,
            log_dir=opt_dir
        )
        report_results(result, f"Optimizer: {opt}")
        all_results["dim4_optimizers"][opt] = result

    # DIMENSION 5: LOSS FUNCTION COMPARISON
    print("\n[Dimension 5] Loss Function: CrossEntropy vs Focal Loss")
    print("=" * 50)
    
    all_results["dim5_loss_functions"] = {}
    
    for loss_fn in ["crossentropy", "focal"]:
        print(f"\n  > Training with {loss_fn.upper()} loss...")
        loss_dir = f"{run_dir}/dim5_{loss_fn}"
        result = train_model_b(
            augment=True,
            model_size="medium",
            subset_ratio=1.0,
            optimizer_name="adam",
            loss_function=loss_fn,
            log_dir=loss_dir
        )
        report_results(result, f"Loss Function: {loss_fn}")
        all_results["dim5_loss_functions"][loss_fn] = result

    # GENERATE FIVE DIMENSION COMPARISON PLOTS
    print("\n[Generating 5 comparison plots...]")
    plot_five_dimension_comparison(all_results, run_dir)

    # Save summary
    summary_path = f"{run_dir}/summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Model B Four-Dimensional Analysis ===\n\n")
        for key, results in all_results.items():
            f.write(f"\n{key}:\n")
            f.write(str(results) + "\n")

    print(f"\n>>> All results saved under: {run_dir}/\n")

def report_results(results_dict, title="Model B Results"):
    """
    print a summary of results
    arguments:
        results_dict: dictionary of results
        title: title for the report
    returns:
        None
    """
    print(f"\n================= {title} =================")
    for metric, value in results_dict.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    print("=" * 50 + "\n")

from datetime import datetime
from .train_a import train_model_a
from .plot_a import plot_four_dimension_comparison

def run_model_a_experiments():
    """
    Run four-dimensional experiments for Model A:
    Dimension 1: Pre-processing (raw vs processed)
    Dimension 2: Augmentation (processed: no aug vs with aug)
    Dimension 3: Model capacity (processed+aug: different C values)
    Dimension 4: Training budget (processed+aug+best_C: different data ratios)
    arguments:
        None
    returns:
        None
    """

    # Create a unique RUN directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"A/logs/run_{timestamp}"
    print(f"\n========= MODEL A EXPERIMENTS: {run_dir} =========\n")

    all_results = {}

    # DIMENSION 1: PRE-PROCESSING COMPARISON
    print("\n[Dimension 1] Pre-processing: Raw vs Processed")
    print("=" * 50)
    
    # Raw features (no augmentation for fair comparison)
    print("\n  > Training on RAW features...")
    raw_dir = f"{run_dir}/dim1_raw"
    results_raw = train_model_a(
        processed=False,
        augment=False,
        subset_ratio=1.0,
        log_dir=raw_dir
    )
    report_results(results_raw, title="RAW Features")
    all_results["dim1_raw"] = results_raw

    # Processed features (no augmentation for fair comparison)
    print("\n  > Training on PROCESSED features (HOG+PCA)...")
    processed_dir = f"{run_dir}/dim1_processed"
    results_processed = train_model_a(
        processed=True,
        augment=False,
        subset_ratio=1.0,
        log_dir=processed_dir
    )
    report_results(results_processed, title="PROCESSED Features (HOG+PCA)")
    all_results["dim1_processed"] = results_processed

    # DIMENSION 2: AUGMENTATION COMPARISON
    print("\n[Dimension 2] Augmentation: Processed (No Aug) vs Processed (With Aug)")
    print("=" * 50)
    
    # Already have processed_no_aug from Dimension 1
    all_results["dim2_processed_no_aug"] = results_processed

    # Processed with augmentation
    print("\n  > Training on PROCESSED features with AUGMENTATION...")
    aug_dir = f"{run_dir}/dim2_processed_aug"
    results_processed_aug = train_model_a(
        processed=True,
        augment=True,
        subset_ratio=1.0,
        log_dir=aug_dir
    )
    report_results(results_processed_aug, title="PROCESSED + AUGMENTED")
    all_results["dim2_processed_aug"] = results_processed_aug

    # Find best C from dimension 2 for subsequent experiments
    best_C = max(results_processed_aug.keys(), 
                 key=lambda c: results_processed_aug[c]["test"]["accuracy"])
    print(f"\n  > Best capacity found: C={best_C}")

    # DIMENSION 3: CAPACITY COMPARISON (using best config)
    print("\n[Dimension 3] Model Capacity: Different C values (Processed+Aug)")
    print("=" * 50)
    print("  > Already trained in Dimension 2")
    all_results["dim3_capacity"] = results_processed_aug

    # DIMENSION 4: TRAINING BUDGET COMPARISON
    print("\n[Dimension 4] Training Budget: Different data ratios (Processed+Aug, C={})".format(best_C))
    print("=" * 50)
    
    budget_ratios = [0.1, 0.3, 0.5, 1.0]
    all_results["dim4_budgets"] = {}
    
    for ratio in budget_ratios:
        print(f"\n  > Training with {ratio*100:.0f}% training data...")
        budget_dir = f"{run_dir}/dim4_budget_{ratio}"
        results_budget = train_model_a(
            processed=True,
            augment=True,
            subset_ratio=ratio,
            # capacity_list=[best_C],  # Use only best C
            log_dir=budget_dir
        )
        report_results(results_budget, title=f"Budget {ratio*100:.0f}%")
        all_results["dim4_budgets"][ratio] = results_budget[best_C]

    # GENERATE FOUR DIMENSION COMPARISON PLOTS
    print("\n[Generating 4 comparison plots...]")
    plot_four_dimension_comparison(all_results, run_dir, best_C)

    # Save summary
    summary_path = f"{run_dir}/summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Model A Four-Dimensional Analysis ===\n\n")
        for key, results in all_results.items():
            f.write(f"\n{key}:\n")
            f.write(str(results) + "\n")

    print(f"\n>>> All results saved under: {run_dir}/\n")


def report_results(results_dict, title="Model A Results"):
    """
    print summary of results
    inputs:
        results_dict: dictionary from train_model_a()
        title: title for the report
    returns: None
    """
    print(f"\n================= {title} =================")
    for C, info in results_dict.items():
        print(f"\n---- Capacity C={C} ----")
        print(f"  Validation - Acc: {info['val']['accuracy']:.4f}, F1: {info['val']['f1']:.4f}")
        print(f"  Test      - Acc: {info['test']['accuracy']:.4f}, F1: {info['test']['f1']:.4f}")
    print("=" * 50 + "\n")
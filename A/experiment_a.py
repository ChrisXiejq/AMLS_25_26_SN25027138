from datetime import datetime
from .train_a import train_model_a

def run_model_a_experiments():
    """
    Run a series of experiments for Model A, logging results to unique directories.
    experiments include:
    1. RAW features
    2. PROCESSED features with HOG + PCA
    3. Data augmentation
    4. Different training budgets
    """

    # Create a unique RUN directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"model_a/logs/run_{timestamp}"
    print(f"\n========= MODEL A EXPERIMENTS: {run_dir} =========\n")

    # ----- Experiment 1: RAW -----
    raw_dir = f"{run_dir}/raw"
    results_raw = train_model_a(
        processed=False,
        augment=False,
        subset_ratio=0.1,
        log_dir=raw_dir
    )
    report_results(results_raw, title="RAW Features Results")

    # ----- Experiment 2: PROCESSED + HOG/PCA -----
    processed_dir = f"{run_dir}/processed"
    results_processed = train_model_a(
        processed=True,
        augment=False,
        subset_ratio=0.1,
        log_dir=processed_dir
    )
    report_results(results_processed, title="Processed Features Results")

    # ----- Experiment 3: Augmentation -----
    augment_dir = f"{run_dir}/augment"
    results_aug = train_model_a(
        processed=True,
        augment=True,
        subset_ratio=0.1,
        log_dir=augment_dir
    )
    report_results(results_aug, title="Augmented Data Results")

    # ----- Experiment 4: Different Training Budget -----
    budget_dir = f"{run_dir}/budget_0.3"
    results_budget = train_model_a(
        processed=True,
        augment=True,
        subset_ratio=0.3,
        log_dir=budget_dir
    )
    report_results(results_budget, title="Training Budget 0.3 Results")

    # Save summary
    summary_path = f"{run_dir}/summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Model A Experiment Summary ===\n\n")
        f.write(str(results_raw) + "\n\n")
        f.write(str(results_processed) + "\n\n")
        f.write(str(results_aug) + "\n\n")
        f.write(str(results_budget) + "\n\n")

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
        print("Validation:")
        print(info["val"])
        print("Test:")
        print(info["test"])
    print("====================================================\n")
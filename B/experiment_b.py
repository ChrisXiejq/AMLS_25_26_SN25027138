import os
from datetime import datetime
from .train_b import train_model_b

def run_model_b_experiments():
    """
    Run a series of experiments for Model B, logging results to unique directories.
    experiments include:
    1. Model capacity variations
    2. Data augmentation effects
    3. Training budget impacts
    4. Different optimizers
    """

    # Create a unique RUN directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"B/logs/run_{timestamp}"

    # Create a TXT file for logging
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "results.txt")
    log_file = open(log_path, "w")

    # ------------------------------
    # Helper: print to screen + file
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    # Start logging
    log("=============== MODEL B EXPERIMENTS ===============")
    log(f"Run directory: {run_dir}")
    log("Timestamp: " + timestamp)
    log("===================================================")

    log("\n=== EXPERIMENT 1: Model Capacity ===")
    cap_dir = run_dir + "/capacity"

    cap_small  = train_model_b(model_size="small",  log_dir=cap_dir+"/small")
    log(f"small:  {cap_small}")

    cap_medium = train_model_b(model_size="medium", log_dir=cap_dir+"/medium")
    log(f"medium: {cap_medium}")

    cap_large  = train_model_b(model_size="large",  log_dir=cap_dir+"/large")
    log(f"large:  {cap_large}")

    log("\n=== EXPERIMENT 2: Data Augmentation ===")
    aug_dir = run_dir + "/augmentation"

    aug_off = train_model_b(augment=False, model_size="medium", log_dir=aug_dir+"/no_augmentation")
    log(f"no augmentation: {aug_off}")

    aug_on  = train_model_b(augment=True,  model_size="medium", log_dir=aug_dir+"/with_augmentation")
    log(f"with augmentation: {aug_on}")

    log("\n=== EXPERIMENT 3: Training Budget ===")
    bud_30  = train_model_b(subset_ratio=0.3, model_size="medium", log_dir=run_dir + "/budget/30_percent_budget")
    log(f"30% budget: {bud_30}")

    bud_100 = train_model_b(subset_ratio=1.0, model_size="medium", log_dir=run_dir + "/budget/full_budget")
    log(f"100% budget: {bud_100}")

    log("\n=== EXPERIMENT 4: Optimizer Comparison ===")
    for opt in ["sgd", "adam", "rmsprop"]:
        log(f"\n>>> Running optimizer = {opt}")
        result = train_model_b(
            optimizer_name=opt,
            model_size="medium",
            epochs=20,
            augment=True,
            subset_ratio=1.0,
            log_dir=run_dir + f"/optimizer/{opt}"
        )
        log(f"{opt}: {result}")

    log("\n=== ALL EXPERIMENTS DONE ===")

    log_file.close()
    print(f"\nAll results have been saved to:\n{log_path}")

def report_results(results_dict):
    """
    print a summary of results
    arguments:
        results_dict: dictionary of results
    returns:
        None
    """
    print("\n================= MODEL B SUMMARY =================")
    for metric, value in results_dict.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print("====================================================\n")

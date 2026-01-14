import os
from A.download_dataset_for_test import prepare_breastmnist_dataset
from A.experiment_a import run_model_a_experiments, run_model_a_processed_only
from B.experiment_b import run_model_b_experiments


if __name__ == "__main__":

    dataset_path = os.path.join("Datasets", "BreastMNIST", "breastmnist.npz")
    if not os.path.isfile(dataset_path):
        print(f"[Data] breastmnist.npz not found at {dataset_path}, downloading...")
        prepare_breastmnist_dataset()
    else:
        print(f"[Data] Found dataset at {dataset_path}")

    # run_model_a_processed_only()
    run_model_a_experiments()
    run_model_b_experiments()
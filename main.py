from model_a.download_dataset_for_test import prepare_breastmnist_dataset
from model_a.eval_a import run_model_a_experiments
from model_b.eval_b import run_model_b_experiments


if __name__ == "__main__":

    prepare_breastmnist_dataset()
    run_model_a_experiments()
    run_model_b_experiments()
import os
import sys
from pathlib import Path

from model_a.download_dataset_for_test import prepare_breastmnist_dataset as download_dataset_for_test
from model_a.data_loader_a import load_numpy
from model_b.data_loader_b import get_dataloaders
from model_a.eval_a import run_model_a_experiments

from model_b.eval_b import run_model_b_experiments

def check_model_a_data():
    print("=== Model A: numpy data ===")
    X_train, y_train = load_numpy("train")
    print("Train:", X_train.shape, y_train.shape)

def check_model_b_data():
    print("=== Model B: torch dataloaders ===")
    train_loader, val_loader, _ = get_dataloaders()
    xb, yb = next(iter(train_loader))
    print("Batch:", xb.shape, yb.shape)


if __name__ == "__main__":

    # download_dataset_for_test()
    # check_model_a_data()
    # check_model_b_data()
    # print("Data check finished.")
    run_model_a_experiments()
    # run_model_b_experiments()


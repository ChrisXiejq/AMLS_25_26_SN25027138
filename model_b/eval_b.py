import os
import numpy as np
from .train_b import train_model_b


def run_model_b_experiments():
    os.makedirs("outputs/model_b", exist_ok=True)

    print("\n=== EXPERIMENT 1: Model Capacity ===")
    # cap_small  = train_model_b(model_size="small")
    # cap_medium = train_model_b(model_size="medium")
    # cap_large  = train_model_b(model_size="large")

    # print("\n=== EXPERIMENT 2: Data Augmentation ===")
    # aug_off = train_model_b(augment=False, model_size="medium")
    aug_on  = train_model_b(augment=True,  model_size="medium")

    # print("\n=== EXPERIMENT 3: Training Budget ===")
    # bud_30 = train_model_b(subset_ratio=0.3)
    # bud_100 = train_model_b(subset_ratio=1.0)

    print("\n=== ALL EXPERIMENTS DONE ===")

def report_results(results_dict):
    """
    输入 train_model_b() 的返回结果
    格式化输出，用于写报告
    """
    print("\n================= MODEL B SUMMARY =================")
    for metric, value in results_dict.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print("====================================================\n")

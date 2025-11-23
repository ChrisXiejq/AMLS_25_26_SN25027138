from .train_a import evaluate_model


def report_results(results_dict):
    """
    输入 train_model_a() 的返回结果
    格式化输出，用于写报告
    """
    print("\n================= MODEL A SUMMARY =================")
    for C, info in results_dict.items():
        print(f"\n---- Capacity C={C} ----")
        print("Validation:")
        print(info["val"])
        print("Test:")
        print(info["test"])
    print("====================================================\n")

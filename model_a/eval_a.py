from .train_a import train_model_a
from .eval_a import report_results

def run_model_a_experiments():
    print("\n========= Running Model A Experiments (SVM + PCA) =========\n")

    results_raw = train_model_a(augment=False)

    report_results(results_raw, title="Model A Results raw pixels")

    results_with_pca = train_model_a(
        use_pca=True,
        pca_dim=50,
        capacity_list=[0.1, 1, 10],
        # subset_ratio=1.0,  # set to 1.0 for full training
        subset_ratio=0.1,  # set to 0.1 for quick tests
    )

    report_results(results_with_pca, title="Model A Results with PCA")

    results_hog = train_model_a(
        use_pca=False,
        capacity_list=[0.1, 1, 10],
        augment=True,
        subset_ratio=0.1,  # set to 0.1 for quick tests
    )

    report_results(results_hog, title="Model A Results with HOG")

    results_budget = train_model_a(
        use_pca=True,
        pca_dim=50,
        capacity_list=[0.1, 1, 10],
        subset_ratio=0.3,  # training budget
        augment=False
    )

    report_results(results_budget, title="Model A Results with Training Budget")

def report_results(results_dict, title="Model A Results"):
    """
    输入 train_model_a() 的返回结果
    格式化输出，用于写报告
    """
    print(f"\n================= {title} =================")
    for C, info in results_dict.items():
        print(f"\n---- Capacity C={C} ----")
        print("Validation:")
        print(info["val"])
        print("Test:")
        print(info["test"])
    print("====================================================\n")

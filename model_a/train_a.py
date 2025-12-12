# Code/model_a/train_a.py
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

from .augmentation import augment_image
from .data_loader_a import load_numpy
from .plot_a import plot_capacity_performance, plot_conf_matrix, plot_train_budget, ensure_dir
from .data_process_a import prepare_features

def build_model(capacity=1.0):
    """Build SVM with StandardScaler."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=capacity, kernel="rbf", gamma="scale"))
    ])

def train_model_a(
    processed=False,
    pca_dim=50,
    capacity_list=[0.1, 1, 10], # different C values for SVM
    subset_ratio=1.0,     # training budget
    augment=True,         # enable data augmentation
    log_dir="model_a/logs"
):
    """
    Train Model A: SVM with optional PCA and HOG feature augmentation.
    arguments:
        use_pca: whether to use PCA for dimensionality reduction
        pca_dim: target dimension for PCA
        capacity_list: list of C values for SVM to train
        subset_ratio: ratio of training data to use (for training budget experiments)
        augment: whether to use HOG feature augmentation
    returns:
        results: dictionary of results for each capacity
    """
    ensure_dir(log_dir)
    log_text = []
    log_text.append(f"===== Experiment Log =====\n")
    log_text.append(f"processed={processed}\n")
    log_text.append(f"pca_dim={pca_dim}\n")
    log_text.append(f"capacity_list={capacity_list}\n")
    log_text.append(f"subset_ratio={subset_ratio}\n")
    log_text.append(f"augment={augment}\n\n")

    # Load numpy data
    # Load raw images without flattening for potential HOG feature extraction
    X_train, y_train = load_numpy("train", flatten=False)
    X_val, y_val = load_numpy("val", flatten=False)
    X_test, y_test = load_numpy("test", flatten=False)

    if augment:
        print("Applying data augmentation to training images")
        X_train = np.array([augment_image(img) for img in X_train])
        log_text.append("Applied augmentation.\n")

    # Feature selection
    X_train, X_val, X_test = prepare_features(
        X_train, X_val, X_test,
        mode="processed" if processed else "raw",
        pca_dim=pca_dim
    )


    # Training budget
    if subset_ratio < 1.0:
        n = int(len(X_train) * subset_ratio)
        print(f"[Training Budget] Using {n}/{len(X_train)} samples")
        X_train = X_train[:n]
        y_train = y_train[:n]
        log_text.append(f"Training budget used = {n}\n")

    results = {}

    model_a_dir = os.path.dirname(os.path.abspath(__file__))

    # Train multiple capacities
    for C in capacity_list:
        model_path = os.path.join(
            model_a_dir,
            "saved_models",
            f"modelA_C{C}_processed{processed}_augment{augment}_budget{subset_ratio}.pkl"
        )
        ensure_dir(os.path.dirname(model_path))
        log_text.append(f"\n===== Training SVM (C={C}) =====\n")
        model = build_model(capacity=C)
        model.fit(X_train, y_train)

        if os.path.exists(model_path):
            print(f"\n>>> Pretrained model found. Loading {model_path}")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        else:
            print(f"\n===== Training SVM (C={C}) =====")
            model = build_model(capacity=C)
            model.fit(X_train, y_train)

            # save model
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f">>> Saved trained model to {model_path}")

        res_val = evaluate_model(model, X_val, y_val)
        res_test = evaluate_model(model, X_test, y_test)

        results[C] = {"val": res_val, "test": res_test, "model": model}
        log_text.append(f"[Val ] {res_val}\n")
        log_text.append(f"[Test] {res_test}\n")

        print(f"[Val] acc={res_val['accuracy']:.4f}, f1={res_val['f1']:.4f}")
        print(f"[Test] acc={res_test['accuracy']:.4f}, f1={res_test['f1']:.4f}")

    # Visualization
    plot_capacity_performance(results, log_dir)

    best_C = max(results.keys(), key=lambda c: results[c]["test"]["accuracy"])
    plot_conf_matrix(results, best_C, X_test, y_test, log_dir)

    write_log(os.path.join(log_dir, "experiment_log.txt"), "".join(log_text))

    return results


def evaluate_model(model, X_test, y_test):
    """Return metrics dictionary."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

def write_log(filepath, content):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w") as f:
        f.write(content)
    print(f"[LOG] Saved → {filepath}")

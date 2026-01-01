# Code/model_a/train_a.py
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from .augmentation import augment_image
from .data_loader_a import load_numpy
from .plot_a import (
    plot_capacity_performance,
    plot_conf_matrix,
    plot_train_budget,
    ensure_dir,
    plot_roc_curve,
    plot_learning_curve_data,
    plot_feature_projection,
)
from .data_process_a import prepare_features

def build_model(capacity=1.0):
    """Build SVM with StandardScaler.
    args:
        capacity: C parameter for SVM
    returns:        sklearn Pipeline model
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=capacity, kernel="rbf", gamma="scale"))
    ])

def train_model_a(
    processed=False,
    pca_dim=50,
    capacity_list=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
    subset_ratio=1.0,
    augment=True,    
    log_dir="A/logs"
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
    log_text.append("===== Experiment Log =====\n")
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
        log_text.append(f"\n===== Training Incremental SVM (C={C}) =====\n")
        
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

    # ROC-AUC for the best model on test set
    best_model = results[best_C]["model"]
    scores = best_model.decision_function(X_test)
    plot_roc_curve(y_test, scores, os.path.join(log_dir, f"roc_curve_bestC{best_C}.png"),
                   title=f"ROC Curve (C={best_C}, processed={processed}, aug={augment}, budget={subset_ratio})")

    write_log(os.path.join(log_dir, "experiment_log.txt"), "".join(log_text))

    return results


def evaluate_model(model, X_test, y_test):
    """Return metrics dictionary.
    args:
        model: trained sklearn model
        X_test: test features
        y_test: test labels
    returns:        dict of metrics
    """
    y_pred = model.predict(X_test)
    scores = model.decision_function(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, scores),
    }

def write_log(filepath, content):
    """Write content to a log file.
    args:
        filepath: path to the log file
        content: string content to write
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w") as f:
        f.write(content)
    print(f"[LOG] Saved → {filepath}")


def generate_learning_curve(
    processed=True,
    pca_dim=50,
    capacity=1.0,
    augment=True,
    ratios=None,
    log_dir="A/logs"
):
    """Train SVM on increasing data fractions to plot a learning curve.
    args:
        processed: whether to use processed features
        pca_dim: PCA dimension if processed 
        capacity: C parameter for SVM
        augment: whether to use data augmentation
        ratios: list of data size ratios to use
        log_dir: directory to save logs and plots
    returns:
        dict with ratios, accuracies, and f1-scores
    """
    if ratios is None:
        ratios = [0.1, 0.3, 0.5, 0.7, 1.0]

    ensure_dir(log_dir)

    # Load data
    X_train, y_train = load_numpy("train", flatten=False)
    X_val, y_val = load_numpy("val", flatten=False)
    X_test, y_test = load_numpy("test", flatten=False)

    if augment:
        print("Applying data augmentation to training images")
        X_train = np.array([augment_image(img) for img in X_train])

    # Feature processing
    X_train_full, X_val_proc, X_test_proc = prepare_features(
        X_train, X_val, X_test,
        mode="processed" if processed else "raw",
        pca_dim=pca_dim
    )

    accs, f1s = [], []

    for r in ratios:
        n = max(1, int(len(X_train_full) * r))
        print(f"[Learning Curve] Using {n}/{len(X_train_full)} samples (ratio={r})")

        X_sub = X_train_full[:n]
        y_sub = y_train[:n]

        model = build_model(capacity=capacity)
        model.fit(X_sub, y_sub)

        res_test = evaluate_model(model, X_test_proc, y_test)
        accs.append(res_test["accuracy"])
        f1s.append(res_test["f1"])

    plot_learning_curve_data(ratios, accs, f1s, log_dir, filename=f"learning_curve_C{capacity}.png")
    print(f"  ✓ Saved learning curve to {log_dir}/learning_curve_C{capacity}.png")

    return {
        "ratios": ratios,
        "accuracy": accs,
        "f1": f1s,
    }


def generate_feature_projection(
    processed=True,
    pca_dim=50,
    augment=False,
    subset_ratio=1.0,
    log_dir="A/logs",
):
    """Project features to 2D (PCA) for visualization.
    args:
        processed: whether to use processed features
        pca_dim: PCA dimension if processed 
        augment: whether to use data augmentation
        subset_ratio: ratio of training data to use
        log_dir: directory to save plots
    returns:
        X_2d: 2D projected features
        y: corresponding labels
    """
    ensure_dir(log_dir)

    X_train, y_train = load_numpy("train", flatten=False)
    X_val, y_val = load_numpy("val", flatten=False)
    X_test, y_test = load_numpy("test", flatten=False)

    if augment:
        print("Applying data augmentation to training images")
        X_train = np.array([augment_image(img) for img in X_train])

    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)

    X_proc, _, _ = prepare_features(
        X_all, X_val, X_test,
        mode="processed" if processed else "raw",
        pca_dim=pca_dim,
    )

    if subset_ratio < 1.0:
        n = max(1, int(len(X_proc) * subset_ratio))
        X_proc = X_proc[:n]
        y_all = y_all[:n]
        print(f"[Feature Viz] Using subset {n}/{len(X_proc)}")

    reducer = PCA(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(X_proc)

    plot_feature_projection(
        X_2d,
        y_all,
        save_path=os.path.join(log_dir, f"feature_projection_processed{processed}_aug{augment}.png"),
        title="Feature Projection (PCA 2D)",
    )

    return X_2d, y_all

# Code/model_a/train_a.py
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from skimage.feature import hog

from .data_loader_a import load_numpy
from .plot_a import plot_capacity_performance, plot_conf_matrix, plot_train_budget, ensure_dir


def extract_hog_features(X):
    """Apply HOG feature extraction to each 28x28 image (Not flattened)."""
    hog_features = []
    for img in X.reshape(-1, 28, 28):
        feat = hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2), orientations=9)
        hog_features.append(feat)
    return np.array(hog_features)


def build_model(capacity=1.0, use_pca=False, pca_dim=50):
    """
    Build Model A: SVM + optional PCA
    """
    steps = []

    steps.append(("scaler", StandardScaler()))

    if use_pca:
        steps.append(("pca", PCA(n_components=pca_dim)))

    svm = SVC(C=capacity, kernel="rbf", gamma="scale")
    steps.append(("svm", svm))

    return Pipeline(steps)


def evaluate_model(model, X_test, y_test):
    """Return metrics dictionary."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def train_model_a(
    use_pca=True,
    pca_dim=50,
    capacity_list=[0.1, 1, 10], # different C values for SVM
    subset_ratio=1.0,     # training budget
    hog=False,         # enable HOG augmentation
    augment=True         # enable data augmentation
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

    # Load numpy data
    # Load raw images without flattening for potential HOG feature extraction
    X_train, y_train = load_numpy("train", flatten=False)
    X_val, y_val = load_numpy("val", flatten=False)
    X_test, y_test = load_numpy("test", flatten=False)

    if augment:
        print("Applying data augmentation to training images...")
        from .augmentation import augment_image
        X_train_augmented = []
        for img in X_train:
            X_train_augmented.append(augment_image(img))
        X_train = np.array(X_train_augmented)

    # HOG
    if hog:
        print("Using HOG feature augmentation...")
        X_train = extract_hog_features(X_train)
        X_val   = extract_hog_features(X_val)
        X_test  = extract_hog_features(X_test)
    else:
        # flatten images
        print("Using raw pixel features...")
        X_train = X_train.reshape(len(X_train), -1)
        X_val   = X_val.reshape(len(X_val), -1)
        X_test  = X_test.reshape(len(X_test), -1)

    # Training budget
    if subset_ratio < 1.0:
        n = int(len(X_train) * subset_ratio)
        print(f"[Training Budget] Using {n}/{len(X_train)} samples")
        X_train = X_train[:n]
        y_train = y_train[:n]

    results = {}

    model_a_dir = os.path.dirname(os.path.abspath(__file__))

    # Train multiple capacities
    for C in capacity_list:
        model_path = os.path.join(
            model_a_dir,
            "saved_models",
            f"modelA_C{C}_pca{use_pca}_{pca_dim}_hog{hog}_budget{subset_ratio}.pkl"
        )
        ensure_dir(os.path.dirname(model_path))
        print(f"\n===== Training SVM (C={C}) =====")
        model = build_model(capacity=C, use_pca=use_pca, pca_dim=pca_dim)

        model.fit(X_train, y_train)

        if os.path.exists(model_path):
            print(f"\n>>> Pretrained model found. Loading {model_path}")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        else:
            print(f"\n===== Training SVM (C={C}) =====")
            model = build_model(capacity=C, use_pca=use_pca, pca_dim=pca_dim)
            model.fit(X_train, y_train)

            # save model
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f">>> Saved trained model to {model_path}")

        res_val = evaluate_model(model, X_val, y_val)
        res_test = evaluate_model(model, X_test, y_test)

        results[C] = {"val": res_val, "test": res_test, "model": model}

        print(f"[Val] acc={res_val['accuracy']:.4f}, f1={res_val['f1']:.4f}")
        print(f"[Test] acc={res_test['accuracy']:.4f}, f1={res_test['f1']:.4f}")

    # Visualization
    output_dir = "outputs/model_a"
    ensure_dir(output_dir)

    plot_capacity_performance(results, output_dir)

    best_C = max(results.keys(), key=lambda c: results[c]["test"]["accuracy"])
    plot_conf_matrix(results, best_C, X_test, y_test, output_dir)

    return results

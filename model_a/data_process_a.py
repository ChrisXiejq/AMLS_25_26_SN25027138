# model_a/data_process_a.py
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA

def extract_hog_features(X):
    """
    Extract HOG features from images.
    Args:
        X (numpy.ndarray): Input image array of shape (N, 28, 28).
    Returns:
        numpy.ndarray: HOG feature array.
    """
    hog_features = []
    for img in X.reshape(-1, 28, 28):
        feat = hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2), orientations=9)
        hog_features.append(feat)
    return np.array(hog_features)

def prepare_features(X_train, X_val, X_test, mode="raw", pca_dim=50):
    """
    Prepare features based on the specified
    mode: "raw" for raw pixel features,
          "processed" for HOG + PCA features.
    Args:
        X_train (numpy.ndarray): Training images.
        X_val (numpy.ndarray): Validation images.
        X_test (numpy.ndarray): Test images.
        mode (str): Feature mode.
        pca_dim (int): Target dimension for PCA.
    Returns:
        X_train, X_val, X_test: Transformed feature arrays.
    """
    if mode == "raw":
        print("Using RAW pixel features...")
        X_train = X_train.reshape(len(X_train), -1)
        X_val   = X_val.reshape(len(X_val), -1)
        X_test  = X_test.reshape(len(X_test), -1)
        return X_train, X_val, X_test

    elif mode == "processed":
        print("Using PROCESSED features: HOG + PCA...")

        # HOG first
        X_train_hog = extract_hog_features(X_train)
        X_val_hog   = extract_hog_features(X_val)
        X_test_hog  = extract_hog_features(X_test)

        # PCA
        pca = PCA(n_components=pca_dim)
        X_train_pca = pca.fit_transform(X_train_hog)
        X_val_pca   = pca.transform(X_val_hog)
        X_test_pca  = pca.transform(X_test_hog)

        return X_train_pca, X_val_pca, X_test_pca

    else:
        raise ValueError("Unknown feature mode!")

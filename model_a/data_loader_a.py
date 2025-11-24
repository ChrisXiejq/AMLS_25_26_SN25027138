import os
import numpy as np
from medmnist import BreastMNIST


def get_dataset_root():
    """Return path to Datasets/BreastMNIST folder."""
    dataset_root = os.path.join("./", "Datasets", "BreastMNIST")
    os.makedirs(dataset_root, exist_ok=True)
    return dataset_root


def load_numpy(split="train", flatten=True, normalize=True):
    """
    Provide numpy data for traditional models:
    - split: 'train', 'val', 'test'
    - flatten: whether to flatten (N, 784)
    - normalize: whether to divide by 255
    """
    assert split in ["train", "val", "test"]

    root = get_dataset_root()

    ds = BreastMNIST(
        split=split,
        root=root,
        download=False,
        transform=None  # Keep the original numpy format
    )

    X = ds.imgs.astype(np.float32)
    y = ds.labels.squeeze()

    # Some versions may have shape (N, 28, 28, 3), take grayscale
    if X.ndim == 4:
        X = X[..., 0]

    if normalize:
        X /= 255.0

    if flatten:
        X = X.reshape((X.shape[0], -1))

    return X, y
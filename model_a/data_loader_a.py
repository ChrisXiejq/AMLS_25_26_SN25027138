import os
import numpy as np
from medmnist import BreastMNIST


def get_dataset_root():
    """
    Get the root directory for the BreastMNIST dataset.
    Returns:
        str: Path to the dataset root directory.
    """
    dataset_root = os.path.join("./", "Datasets", "BreastMNIST")
    os.makedirs(dataset_root, exist_ok=True)
    return dataset_root


def load_numpy(split="train", flatten=True, normalize=True):
    """
    Provide numpy data for traditional models:
    - split: 'train', 'val', 'test'
    - flatten: whether to flatten (N, 784)
    - normalize: whether to divide by 255
    Returns:
        X (numpy.ndarray): Image data.
        y (numpy.ndarray): Labels.
    """
    assert split in ["train", "val", "test"]

    root = get_dataset_root()

    ds = BreastMNIST(
        split=split,
        root=root,
        download=False,
        transform=None
    )

    X = ds.imgs.astype(np.float32)
    y = ds.labels.squeeze()

    if X.ndim == 4:
        X = X[..., 0]

    if normalize:
        X /= 255.0

    if flatten:
        X = X.reshape((X.shape[0], -1))

    return X, y
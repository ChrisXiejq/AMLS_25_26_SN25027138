# Code/model_a/data_loader.py

import os
import numpy as np
from medmnist import BreastMNIST


def get_breastmnist_root():
    """
    获取数据目录：项目根目录下的 Datasets 文件夹
    """
    this_file = os.path.abspath(__file__)
    model_dir = os.path.dirname(this_file)            # .../Code/model_a
    code_dir = os.path.dirname(model_dir)             # .../Code
    project_root = os.path.dirname(code_dir)          # ...
    datasets_root = os.path.join(project_root, "Datasets")

    os.makedirs(datasets_root, exist_ok=True)
    return datasets_root


def load_numpy(split="train", flatten=True, normalize=True):
    """
    为传统模型提供 numpy 数据:
    - split: 'train', 'val', 'test'
    - flatten: 是否展平 (N, 784)
    - normalize: 是否除以 255
    """
    assert split in ["train", "val", "test"]

    root = get_breastmnist_root()

    ds = BreastMNIST(
        split=split,
        root=root,
        download=True,
        transform=None  # 保留原始 numpy 格式
    )

    X = ds.imgs.astype(np.float32)
    y = ds.labels.squeeze()

    # 部分版本可能为 (N, 28, 28, 3)，取灰度
    if X.ndim == 4:
        X = X[..., 0]

    if normalize:
        X /= 255.0

    if flatten:
        X = X.reshape((X.shape[0], -1))

    return X, y

# Code/model_b/data_loader.py

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BreastMNIST

def target_to_int(y):
    return int(y)

def get_breastmnist_root():
    """
    返回 Datasets 目录路径
    """
    this_file = os.path.abspath(__file__)
    model_dir = os.path.dirname(this_file)           # .../Code/model_b
    code_dir = os.path.dirname(model_dir)            # .../Code
    project_root = os.path.dirname(code_dir)
    datasets_root = os.path.join(project_root, "Datasets")

    os.makedirs(datasets_root, exist_ok=True)
    return datasets_root

def get_transform(augment=False):
    """
    CNN 使用的 transform，包含可选增强
    """
    ops = []

    if augment:
        ops.extend([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
        ])

    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    return transforms.Compose(ops)


def get_datasets(augment_train=True):
    """
    返回 train/val/test Dataset
    """
    root = get_breastmnist_root()

    train_ds = BreastMNIST(
        split="train",
        root=root,
        download=True,
        transform=get_transform(augment=augment_train),
        target_transform=target_to_int
    )

    val_ds = BreastMNIST(
        split="val",
        root=root,
        download=True,
        transform=get_transform(augment=False),
        target_transform=target_to_int
    )

    test_ds = BreastMNIST(
        split="test",
        root=root,
        download=True,
        transform=get_transform(augment=False),
        target_transform=target_to_int
    )

    return train_ds, val_ds, test_ds


def get_dataloaders(batch_size=64, augment_train=True, num_workers=2):
    """
    返回 DataLoader
    """
    train_ds, val_ds, test_ds = get_datasets(augment_train=augment_train)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BreastMNIST

def target_to_int(y):
    return int(y)

def get_dataset_root():
    """Return path to Datasets/BreastMNIST folder."""
    dataset_root = os.path.join("./", "Datasets", "BreastMNIST")
    os.makedirs(dataset_root, exist_ok=True)
    return dataset_root

def get_transform(augment=False):
    """
    CNN transform with optional augmentation
    Args:
        augment (bool): whether to apply data augmentation
    Returns:
        torchvision.transforms.Compose: the composed transform
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
    returns train/val/test Dataset
    """
    root = get_dataset_root()

    train_ds = BreastMNIST(
        split="train",
        root=root,
        download=False,
        transform=get_transform(augment=augment_train),
        target_transform=target_to_int
    )

    val_ds = BreastMNIST(
        split="val",
        root=root,
        download=False,
        transform=get_transform(augment=False),
        target_transform=target_to_int
    )

    test_ds = BreastMNIST(
        split="test",
        root=root,
        download=False,
        transform=get_transform(augment=False),
        target_transform=target_to_int
    )

    return train_ds, val_ds, test_ds


def get_dataloaders(batch_size=64, augment_train=True, num_workers=2):
    """
    returns train/val/test DataLoaders
    Args:
        batch_size (int): batch size for DataLoaders
        augment_train (bool): whether to apply data augmentation to training set
        num_workers (int): number of subprocesses to use for data loading
    Returns:
        tuple: (train_loader, val_loader, test_loader)
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

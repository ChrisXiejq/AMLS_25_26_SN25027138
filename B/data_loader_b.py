import os
import torch
import random
from torchvision import transforms
import torch.nn.functional as F
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

class GaussianNoiseTransform:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Only works on tensors
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise


def get_transform(augment=False):
    """
    CNN transform with optional augmentation for BreastMNIST (28x28 grayscale)
    Args:
        augment (bool): whether to apply data augmentation
    Returns:
        torchvision.transforms.Compose: composed transform
    """
    pil_ops = []
    tensor_ops = []

    if augment:
        # PIL-based transforms
        pil_ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),  # horizontal flip with 50% chance
            transforms.RandomRotation(degrees=10),    # reduce rotation angle from 15 to 10
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # reduce brightness/contrast variation from 0.15 to 0.1
            transforms.RandomAffine(
                degrees=5,              # reduce affine transform angle from 10 to 5
                translate=(0.03, 0.03), # reduce translation from 0.05 to 0.03
                scale=(0.97, 1.03),     # reduce scaling range from 0.9-1.1 to 0.97-1.03
                shear=3                 # reduce shear from 5 to 3
            ),
        ])

    # Tensor-only transforms
    if augment:
        # Removed ElasticTransform - too aggressive for medical images
        # tensor_ops.append(
        #     transforms.ElasticTransform(alpha=3.0, sigma=5.0)
        # )

        # cutout / random erasing - reduce probability and intensity
        tensor_ops.append(
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.05), ratio=(0.5, 2.0))  # reduce erasing intensity
        )

    # common transforms
    tensor_ops.extend([
        GaussianNoiseTransform(std=0.02),  # 减少噪声 0.03->0.02
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    return transforms.Compose(
        pil_ops
        + [transforms.ToTensor()]   # transforms.ToTensor() must be between pil_ops and tensor_ops
        + tensor_ops
    )

def get_datasets(augment_train=True):
    """
    returns train/val/test Dataset
    Args:
        augment_train (bool): whether to apply data augmentation to training set
    Returns:
        tuple: (train_ds, val_ds, test_ds)
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


def get_dataloaders(batch_size=128, augment_train=True, num_workers=0):
    """
    returns train/val/test DataLoaders
    Args:
        batch_size (int): batch size for DataLoaders (increased to 128 for speed)
        augment_train (bool): whether to apply data augmentation to training set
        num_workers (int): number of subprocesses to use for data loading
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_ds, val_ds, test_ds = get_datasets(augment_train=augment_train)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader

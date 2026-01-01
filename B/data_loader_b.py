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
        # PIL-based transforms - 减少激进程度，更适合医学图像
        pil_ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),  # 保留水平翻转
            transforms.RandomRotation(degrees=10),    # 减少旋转角度 15->10
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 减少亮度/对比度变化 0.15->0.1
            transforms.RandomAffine(
                degrees=5,              # 减少仿射变换角度 10->5
                translate=(0.03, 0.03), # 减少平移 0.05->0.03
                scale=(0.97, 1.03),     # 减少缩放范围
                shear=3                 # 减少剪切 5->3
            ),
        ])

    # Tensor-only transforms
    if augment:
        # 移除 ElasticTransform - 对医学图像过于激进
        # tensor_ops.append(
        #     transforms.ElasticTransform(alpha=3.0, sigma=5.0)
        # )

        # cutout / random erasing - 降低概率和强度
        tensor_ops.append(
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.05), ratio=(0.5, 2.0))  # 降低擦除强度
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

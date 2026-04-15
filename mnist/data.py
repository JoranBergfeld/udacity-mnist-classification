"""MNIST data loading."""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Subset

from .augmentation import (
    MNIST_MEAN,
    MNIST_STANDARD_DEVIATION,
    get_augmentation,
    get_eval_transform,
)
from .config import DataConfig

CLASSES = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

DATA_ROOT = Path("data")


def compute_mean_standard_deviation(dataset):
    """Compute per-channel mean and standard deviation from a torchvision MNIST dataset."""
    data = dataset.data.float() / 255.0  # (N, H, W)
    mean = (float(data.mean()),)
    standard_deviation = (float(data.std()),)
    print(f"  Computed mean: ({mean[0]:.4f},)")
    print(f"  Computed standard deviation: ({standard_deviation[0]:.4f},)")
    print(f"  Default mean: ({MNIST_MEAN[0]:.4f},)")
    print(f"  Default standard deviation: ({MNIST_STANDARD_DEVIATION[0]:.4f},)")
    return mean, standard_deviation


def _pin_memory(device_type):
    return device_type == "cuda"


def _make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def get_train_loader(
    augmentation="none",
    batch_size=128,
    num_workers=0,
    data_dir=DATA_ROOT,
    mean=MNIST_MEAN,
    standard_deviation=MNIST_STANDARD_DEVIATION,
    device_type=None,
):
    transform = get_augmentation(augmentation, mean, standard_deviation)
    train_set = torchvision.datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    return _make_loader(train_set, batch_size, True, num_workers, _pin_memory(device_type))


def get_test_loader(
    batch_size=128,
    num_workers=0,
    data_dir=DATA_ROOT,
    mean=MNIST_MEAN,
    standard_deviation=MNIST_STANDARD_DEVIATION,
    device_type=None,
):
    transform = get_eval_transform(mean, standard_deviation)
    test_set = torchvision.datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)
    return _make_loader(test_set, batch_size, False, num_workers, _pin_memory(device_type))


def get_data_loaders(
    augmentation="none",
    batch_size=128,
    num_workers=0,
    data_dir=DATA_ROOT,
    mean=MNIST_MEAN,
    standard_deviation=MNIST_STANDARD_DEVIATION,
    device_type=None,
):
    train = get_train_loader(augmentation, batch_size, num_workers, data_dir, mean, standard_deviation, device_type)
    test = get_test_loader(batch_size, num_workers, data_dir, mean, standard_deviation, device_type)
    return train, test


def get_loaders_from_config(cfg: DataConfig, seed=42, device_type=None):
    """Return (train, val, test) loaders with an optional seeded train/val split.

    The validation subset uses the deterministic eval transform even though it
    is carved out of the training set, so val never sees train-time augmentation.
    """
    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    train_transform = get_augmentation(cfg.augmentation)
    eval_transform = get_eval_transform()

    train_aug = torchvision.datasets.MNIST(str(DATA_ROOT), train=True, download=True, transform=train_transform)
    test = torchvision.datasets.MNIST(str(DATA_ROOT), train=False, download=True, transform=eval_transform)

    pin = _pin_memory(device_type)

    if cfg.val_split and cfg.val_split > 0:
        train_eval = torchvision.datasets.MNIST(str(DATA_ROOT), train=True, download=False, transform=eval_transform)
        n_total = len(train_aug)
        n_val = int(n_total * cfg.val_split)
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(n_total, generator=generator).tolist()
        val_indices = permutation[:n_val]
        train_indices = permutation[n_val:]
        train_ds = Subset(train_aug, train_indices)
        val_ds = Subset(train_eval, val_indices)
        val_loader = _make_loader(val_ds, cfg.batch_size, False, cfg.num_workers, pin)
    else:
        train_ds = train_aug
        val_loader = None

    train_loader = _make_loader(train_ds, cfg.batch_size, True, cfg.num_workers, pin)
    test_loader = _make_loader(test, cfg.batch_size, False, cfg.num_workers, pin)
    return train_loader, val_loader, test_loader


def get_notebook_loaders(batch_size=128, augmentation="none", num_workers=0):
    """Convenience wrapper for the starter notebook: (train, test), no val split."""
    return get_data_loaders(augmentation=augmentation, batch_size=batch_size, num_workers=num_workers)


def get_sample_batch(loader, n=5):
    """Fetch n images and labels from a loader."""
    images, labels = next(iter(loader))
    return images[:n], labels[:n]

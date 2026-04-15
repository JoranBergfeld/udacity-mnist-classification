"""Augmentation registry for MNIST training transforms.

Every transform in the registry is a full `ToTensor + Normalize(...)` pipeline
with zero or more spatial/tensor augmentations spliced around it. The test
pipeline (no augmentation) is exposed as `get_eval_transform`.
"""

from __future__ import annotations

import torchvision.transforms as transforms

MNIST_MEAN = (0.1307,)
MNIST_STANDARD_DEVIATION = (0.3081,)


def _normalize(mean=MNIST_MEAN, standard_deviation=MNIST_STANDARD_DEVIATION):
    return transforms.Normalize(mean, standard_deviation)


def get_none_transforms(mean=MNIST_MEAN, standard_deviation=MNIST_STANDARD_DEVIATION):
    return transforms.Compose([transforms.ToTensor(), _normalize(mean, standard_deviation)])


def get_affine_transforms(mean=MNIST_MEAN, standard_deviation=MNIST_STANDARD_DEVIATION):
    return transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        _normalize(mean, standard_deviation),
    ])


def get_erasing_transforms(mean=MNIST_MEAN, standard_deviation=MNIST_STANDARD_DEVIATION):
    return transforms.Compose([
        transforms.ToTensor(),
        _normalize(mean, standard_deviation),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def get_randaugment_transforms(mean=MNIST_MEAN, standard_deviation=MNIST_STANDARD_DEVIATION):
    return transforms.Compose([
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        _normalize(mean, standard_deviation),
    ])


def get_affine_erasing_transforms(mean=MNIST_MEAN, standard_deviation=MNIST_STANDARD_DEVIATION):
    return transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        _normalize(mean, standard_deviation),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


AUGMENTATION_REGISTRY = {
    "none": get_none_transforms,
    "affine": get_affine_transforms,
    "erasing": get_erasing_transforms,
    "randaugment": get_randaugment_transforms,
    "affine_erasing": get_affine_erasing_transforms,
}


def get_augmentation(name, mean=MNIST_MEAN, standard_deviation=MNIST_STANDARD_DEVIATION):
    if name not in AUGMENTATION_REGISTRY:
        raise ValueError(f"Unknown augmentation: {name}. Choose from {list(AUGMENTATION_REGISTRY)}")
    return AUGMENTATION_REGISTRY[name](mean, standard_deviation)


def get_eval_transform(mean=MNIST_MEAN, standard_deviation=MNIST_STANDARD_DEVIATION):
    """Deterministic pipeline for validation/test: no augmentation."""
    return get_none_transforms(mean, standard_deviation)

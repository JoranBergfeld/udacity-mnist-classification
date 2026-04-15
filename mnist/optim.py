"""Optimizer and scheduler factories."""

from __future__ import annotations

import torch.optim as optim


OPTIMIZER_NAMES = ["adam", "adamw", "sgd"]
SCHEDULER_NAMES = ["none", "step", "cosine", "onecycle"]


def create_optimizer(model, name, learning_rate, weight_decay=0.0, momentum=0.9, nesterov=False):
    if learning_rate <= 0:
        raise ValueError(f"Learning rate must be positive, got {learning_rate}")
    name = name.lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {name}. Choose from {OPTIMIZER_NAMES}")


def create_scheduler(optimizer, name, epochs, steps_per_epoch=None, **kwargs):
    name = name.lower()
    if name == "none":
        return None
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get("step_size", 10), gamma=kwargs.get("gamma", 0.1)
        )
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs.get("t_max", epochs))
    if name == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("onecycle scheduler requires steps_per_epoch")
        max_lr = kwargs.get("max_lr", optimizer.param_groups[0]["lr"] * 10)
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=kwargs.get("pct_start", 0.3),
        )
    raise ValueError(f"Unknown scheduler: {name}. Choose from {SCHEDULER_NAMES}")


def scheduler_is_batch_level(scheduler):
    return isinstance(scheduler, optim.lr_scheduler.OneCycleLR)

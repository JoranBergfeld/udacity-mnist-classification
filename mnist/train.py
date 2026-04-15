"""Training loop, device detection, and helpers."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .optim import scheduler_is_batch_level


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def get_device():
    """Runtime device detection: CUDA → MPS → CPU, with diagnostics."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return device
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        print("  Using Apple MPS")
        return torch.device("mps")
    print("  Using CPU")
    print(f"  CPU threads: {torch.get_num_threads()}")
    print("  WARNING: Training on CPU will be very slow.")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Label-smoothed NLL (models return log_softmax, so nn.CrossEntropyLoss is not used)
# ---------------------------------------------------------------------------


class _SmoothedNLLLoss(nn.Module):
    """Label-smoothing NLL over log-probability inputs.

    Matches torch.nn.CrossEntropyLoss(label_smoothing=s) convention: uniform
    `s/K` mass across all K classes, `(1-s)` additional mass on the true class.
    """

    def __init__(self, smoothing):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, log_probs, target):
        s = self.smoothing
        return (1.0 - s) * F.nll_loss(log_probs, target) + s * -log_probs.mean(dim=-1).mean()


def make_criterion(label_smoothing=0.0):
    if label_smoothing > 0:
        return _SmoothedNLLLoss(smoothing=label_smoothing)
    return nn.NLLLoss()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = 0
    total_time: float = 0.0

    def as_metrics(self):
        """CIFAR-compatible metrics dict for save_run_metrics()."""
        return {
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "total_time": self.total_time,
        }


def _run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    batch_scheduler = training and scheduler is not None and scheduler_is_batch_level(scheduler)
    loss_sum = torch.zeros((), device=device)
    correct_sum = torch.zeros((), device=device, dtype=torch.long)
    total_n = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if batch_scheduler:
                    scheduler.step()
            batch_size = labels.size(0)
            loss_sum += loss.detach() * batch_size
            correct_sum += (outputs.argmax(1) == labels).sum()
            total_n += batch_size
    return loss_sum.item() / total_n, correct_sum.item() / total_n


def train_model(
    model,
    train_loader,
    epochs,
    optimizer,
    scheduler=None,
    device=None,
    val_loader=None,
    criterion=None,
    label_smoothing=0.0,
    early_stopping_patience=0,
    seed=42,
    on_epoch_end: Callable[[int, dict], None] | None = None,
    progress=False,
):
    if epochs <= 0:
        raise ValueError(f"Epochs must be a positive integer, got {epochs}")
    if device is None:
        device = get_device()
    if criterion is None:
        criterion = make_criterion(label_smoothing)

    set_seed(seed)
    model = model.to(device)

    history = TrainHistory()
    best_state = None
    stale = 0
    start = time.time()

    epoch_iterator = range(1, epochs + 1)
    if progress:
        epoch_iterator = tqdm(epoch_iterator, desc="epochs")

    for epoch in epoch_iterator:
        epoch_start = time.time()
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer, scheduler)
        if val_loader is not None:
            val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = float("nan"), float("nan")

        if scheduler is not None and not scheduler_is_batch_level(scheduler):
            scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.learning_rates.append(current_lr)
        history.epoch_times.append(epoch_time)

        if val_loader is not None and not np.isnan(val_acc) and val_acc > history.best_val_acc:
            history.best_val_acc = val_acc
            history.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        print(
            f"  Epoch {epoch}/{epochs} — "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, "
            f"lr: {current_lr:.2e}, time: {epoch_time:.1f}s"
        )

        if on_epoch_end is not None:
            on_epoch_end(epoch, {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": current_lr,
                "epoch_time": epoch_time,
            })

        if early_stopping_patience and stale >= early_stopping_patience:
            print(f"  Early stopping at epoch {epoch} (no val improvement for {stale} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history.total_time = time.time() - start
    return history

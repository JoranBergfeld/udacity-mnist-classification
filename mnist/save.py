"""Model + run-metrics persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


DEFAULT_MODELS_DIR = Path("models")
DEFAULT_RESULTS_DIR = Path("results")


def save_model(model, name, save_dir=DEFAULT_MODELS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(str(save_dir), f"{name}.pt")
    torch.save(model.state_dict(), path)
    print(f"  Model saved to {path}")
    return path


def load_model(model_class, path, **kwargs):
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def save_run_metrics(run_name, metrics, eval_results, config, save_dir=DEFAULT_RESULTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(str(save_dir), f"{run_name}.json")

    data = {
        "config": config,
        "training": {
            "train_loss": metrics.get("train_loss", []),
            "train_acc": metrics.get("train_acc", []),
            "val_loss": metrics.get("val_loss", []),
            "val_acc": metrics.get("val_acc", []),
            "learning_rates": metrics.get("learning_rates", []),
            "epoch_times": metrics.get("epoch_times", []),
            "best_val_acc": metrics.get("best_val_acc", 0.0),
            "best_epoch": metrics.get("best_epoch", 0),
            "total_time": metrics.get("total_time", 0.0),
        },
        "evaluation": {
            "overall_accuracy": eval_results["overall_accuracy"],
            "per_class_accuracy": eval_results.get("per_class_accuracy", {}),
            "loss": eval_results.get("loss", None),
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"  Metrics saved to {path}")
    return path


def load_run_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_results(results_dir=DEFAULT_RESULTS_DIR):
    results = {}
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return results
    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith(".json"):
            run_name = filename[:-5]
            results[run_name] = load_run_metrics(results_dir / filename)
    return results


def make_run_name(model_name, augmentation, optimizer_name, scheduler_name, reg_preset=None, epochs=None):
    """Deterministic short name matching CIFAR's `{model}_{aug}_{opt}_{sched}[...]` convention."""
    name = f"{model_name}_{augmentation}_{optimizer_name}_{scheduler_name}"
    if reg_preset:
        name += f"_{reg_preset}"
    if epochs is not None and epochs != 15:
        name += f"_{epochs}ep"
    return name

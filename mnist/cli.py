"""Unified CLI for MNIST training + sweeps.

Primary interface is YAML:

    mnist --config configs/baseline_mlp.yaml
    mnist --sweep  configs/sweep_overnight.yaml

Convenience grid over CLI flags (CIFAR style):

    mnist --models mlp simple_cnn --augmentations none affine \\
          --optimizers adam --schedulers cosine --epochs 10

Summarize all completed runs (ranked by test accuracy):

    mnist --summary [--top 20] [--out results/summary.md]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from .analysis import confusion_matrix, per_class_report
from .augmentation import AUGMENTATION_REGISTRY
from .config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    SchedulerConfig,
    TrainingConfig,
    load_config,
)
from .data import get_loaders_from_config
from .evaluate import evaluate_model
from .models import MODEL_REGISTRY, get_model
from .optim import OPTIMIZER_NAMES, SCHEDULER_NAMES, create_optimizer, create_scheduler
from .save import (
    DEFAULT_MODELS_DIR,
    DEFAULT_RESULTS_DIR,
    load_all_results,
    save_model,
    save_run_metrics,
)
from .train import TrainHistory, get_device, train_model


# ---------------------------------------------------------------------------
# Run-name derivation
# ---------------------------------------------------------------------------


def run_name_for(cfg: RunConfig) -> str:
    """Deterministic run name from a resolved RunConfig.

    Shape: `{model}_{aug}_{opt}_{sched}_{reg_token}` where reg_token encodes
    dropout / weight_decay / label_smoothing. Short but identifies the recipe.
    """
    model = cfg.model.name
    aug = cfg.data.augmentation
    opt = cfg.optimizer.name
    sched = cfg.scheduler.name
    dropout = cfg.model.kwargs.get("dropout", "")
    parts = [model, aug, opt, sched]
    reg_bits = []
    if dropout != "":
        reg_bits.append(f"do{_fmt(dropout)}")
    if cfg.optimizer.weight_decay:
        reg_bits.append(f"wd{_fmt(cfg.optimizer.weight_decay)}")
    if cfg.training.label_smoothing:
        reg_bits.append(f"ls{_fmt(cfg.training.label_smoothing)}")
    if reg_bits:
        parts.append("-".join(reg_bits))
    if cfg.training.epochs != 15:
        parts.append(f"{cfg.training.epochs}ep")
    return "_".join(parts)


def _fmt(value) -> str:
    return str(value).replace(".", "p").replace("-", "m")


# ---------------------------------------------------------------------------
# Config resolution helpers
# ---------------------------------------------------------------------------


def config_to_dict(cfg: RunConfig) -> dict:
    return asdict(cfg)


def dict_to_config(data: dict) -> RunConfig:
    return RunConfig.from_dict(data)


def dotted_set(data: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = data
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def dotted_get(data: dict, dotted: str) -> Any:
    cur: Any = data
    for k in dotted.split("."):
        cur = cur[k]
    return cur


def pattern_matches(data: dict, pattern: dict) -> bool:
    try:
        return all(dotted_get(data, k) == v for k, v in pattern.items())
    except (KeyError, TypeError):
        return False


def expand_sweep(sweep: dict) -> list[dict]:
    """Expand a sweep YAML into a list of resolved config dicts.

    Schema:
        base:     <RunConfig fields, applied to every run>
        grid:     {dotted.key: [value, ...], ...}
        exclude:  [{dotted.key: value, ...}, ...]  # skip matching combos
    """
    base = sweep.get("base", {}) or {}
    grid = sweep.get("grid", {}) or {}
    exclude = sweep.get("exclude", []) or []

    if not grid:
        return [copy.deepcopy(base)]

    keys = list(grid.keys())
    value_lists = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]

    resolved: list[dict] = []
    for combination in itertools.product(*value_lists):
        cfg_dict = copy.deepcopy(base)
        for key, value in zip(keys, combination):
            dotted_set(cfg_dict, key, value)
        if any(pattern_matches(cfg_dict, pat) for pat in exclude):
            continue
        resolved.append(cfg_dict)
    return resolved


def build_adhoc_sweep(arguments) -> list[dict]:
    """Build a sweep from CLI flags (CIFAR-style). Used when no YAML is given."""
    base = {
        "name": "adhoc",
        "seed": arguments.seed,
        "data": {
            "batch_size": arguments.batch_size,
            "val_split": arguments.val_split,
            "augmentation": "none",
            "num_workers": arguments.num_workers,
        },
        "model": {"name": "mlp", "kwargs": {"dropout": 0.2}},
        "optimizer": {"name": "adam", "lr": arguments.learning_rate, "weight_decay": 0.0},
        "scheduler": {"name": "cosine", "kwargs": {"t_max": arguments.epochs}},
        "training": {
            "epochs": arguments.epochs,
            "label_smoothing": 0.0,
            "early_stopping_patience": 0,
        },
    }
    grid = {
        "model.name": arguments.models,
        "data.augmentation": arguments.augmentations,
        "optimizer.name": arguments.optimizers,
        "scheduler.name": arguments.schedulers,
    }
    return expand_sweep({"base": base, "grid": grid})


# ---------------------------------------------------------------------------
# Single-run orchestrator
# ---------------------------------------------------------------------------


def run_single(
    cfg: RunConfig,
    device,
    models_dir: Path = DEFAULT_MODELS_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    run_name: str | None = None,
) -> tuple[str, float]:
    run_name = run_name or run_name_for(cfg)
    print(f"\n{'=' * 60}")
    print(f"  Running: {run_name}")
    print(f"{'=' * 60}")
    print(
        f"  model={cfg.model.name}  augmentation={cfg.data.augmentation}  "
        f"optimizer={cfg.optimizer.name}  scheduler={cfg.scheduler.name}  "
        f"epochs={cfg.training.epochs}  batch_size={cfg.data.batch_size}  "
        f"lr={cfg.optimizer.lr}"
    )

    model = get_model(cfg.model.name, **cfg.model.kwargs)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {parameter_count:,}")

    train_loader, val_loader, test_loader = get_loaders_from_config(
        cfg.data, seed=cfg.seed, device_type=device.type
    )
    print(
        f"  Train samples: {len(train_loader.dataset)}  "
        f"Val samples: {len(val_loader.dataset) if val_loader else 0}  "
        f"Test samples: {len(test_loader.dataset)}"
    )

    optimizer = create_optimizer(
        model,
        cfg.optimizer.name,
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        momentum=cfg.optimizer.momentum,
        nesterov=cfg.optimizer.nesterov,
    )
    scheduler = create_scheduler(
        optimizer,
        cfg.scheduler.name,
        epochs=cfg.training.epochs,
        steps_per_epoch=len(train_loader),
        **(cfg.scheduler.kwargs or {}),
    )

    history = train_model(
        model,
        train_loader,
        epochs=cfg.training.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        val_loader=val_loader,
        label_smoothing=cfg.training.label_smoothing,
        early_stopping_patience=cfg.training.early_stopping_patience,
        seed=cfg.seed,
    )

    eval_results = evaluate_model(model, test_loader, device=device)
    print(f"  Test accuracy: {eval_results['overall_accuracy']:.2f}%  "
          f"(loss={eval_results['loss']:.4f})")

    # Per-class + confusion matrix for the run log (analysis module)
    report = per_class_report(eval_results["predictions"], eval_results["ground_truth"])
    top_errors = sorted(report.items(), key=lambda kv: kv[1]["f1"])[:3]
    print("  Hardest digits (lowest F1): "
          + ", ".join(f"{name}:{m['f1']:.3f}" for name, m in top_errors))

    save_model(model, run_name, save_dir=models_dir)

    config_dump = config_to_dict(cfg)
    config_dump["parameters"] = parameter_count
    config_dump["device"] = str(device)

    save_run_metrics(
        run_name,
        history.as_metrics(),
        eval_results,
        config_dump,
        save_dir=results_dir,
    )

    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_name, eval_results["overall_accuracy"]


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def print_summary(results_dir: Path = DEFAULT_RESULTS_DIR, top: int = 20, out: str | None = None):
    results = load_all_results(results_dir)
    if not results:
        print(f"No runs found under {results_dir}/")
        return

    rows = []
    for run_name, payload in results.items():
        cfg = payload.get("config", {})
        eval_block = payload.get("evaluation", {})
        training = payload.get("training", {})
        rows.append({
            "run_name": run_name,
            "test_acc": eval_block.get("overall_accuracy", float("nan")),
            "best_val": training.get("best_val_acc", float("nan")) * 100.0
                if training.get("best_val_acc", 0) <= 1.0 else training.get("best_val_acc", float("nan")),
            "epochs": len(training.get("train_loss", [])),
            "total_time_sec": training.get("total_time", 0.0),
            "model": cfg.get("model", {}).get("name", ""),
            "augmentation": cfg.get("data", {}).get("augmentation", ""),
            "optimizer": cfg.get("optimizer", {}).get("name", ""),
            "scheduler": cfg.get("scheduler", {}).get("name", ""),
            "parameters": cfg.get("parameters", 0),
        })
    rows.sort(key=lambda r: r["test_acc"], reverse=True)

    header = (
        f"{'rank':>4}  {'test_acc':>9}  {'best_val':>9}  {'epochs':>6}  "
        f"{'time(s)':>8}  {'model':<14}  {'aug':<16}  {'opt':<6}  {'sched':<10}  run_name"
    )
    lines = [header, "-" * len(header)]
    for rank, row in enumerate(rows[:top], 1):
        lines.append(
            f"{rank:>4}  {row['test_acc']:>9.2f}  {row['best_val']:>9.2f}  "
            f"{row['epochs']:>6}  {row['total_time_sec']:>8.0f}  "
            f"{row['model']:<14}  {row['augmentation']:<16}  "
            f"{row['optimizer']:<6}  {row['scheduler']:<10}  {row['run_name']}"
        )
    print(f"\n{len(rows)} runs total — top {min(top, len(rows))} by test accuracy:\n")
    for line in lines:
        print(line)

    if out:
        out_path = Path(out)
        md_lines = [
            f"# Run summary ({len(rows)} runs)\n",
            "| rank | test_acc | best_val | epochs | time(s) | model | aug | opt | sched | run_name |",
            "|---:|---:|---:|---:|---:|:---|:---|:---|:---|:---|",
        ]
        for rank, row in enumerate(rows[:top], 1):
            md_lines.append(
                f"| {rank} | {row['test_acc']:.2f} | {row['best_val']:.2f} | "
                f"{row['epochs']} | {row['total_time_sec']:.0f} | "
                f"{row['model']} | {row['augmentation']} | {row['optimizer']} | "
                f"{row['scheduler']} | {row['run_name']} |"
            )
        out_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print(f"\nWrote {out}")


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def run_sweep(
    sweep_dicts: list[dict],
    device,
    models_dir: Path = DEFAULT_MODELS_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    dry_run: bool = False,
):
    print(f"Planned runs: {len(sweep_dicts)}")

    configs_with_names: list[tuple[dict, RunConfig, str]] = []
    for cfg_dict in sweep_dicts:
        cfg = dict_to_config(cfg_dict)
        run_name = run_name_for(cfg)
        configs_with_names.append((cfg_dict, cfg, run_name))

    for index, (_, _, run_name) in enumerate(configs_with_names, 1):
        print(f"  [{index:3d}] {run_name}")

    if dry_run:
        return []

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in Path(results_dir).glob("*.json")}

    sweep_start = time.time()
    done = 0
    skipped = 0
    summary: list[tuple[str, float]] = []

    for index, (_, cfg, run_name) in enumerate(configs_with_names, 1):
        if run_name in existing:
            skipped += 1
            print(f"\n[{index}/{len(configs_with_names)}] SKIP (exists): {run_name}")
            continue
        print(f"\n[{index}/{len(configs_with_names)}] RUN: {run_name}")
        try:
            name, accuracy = run_single(cfg, device, models_dir, results_dir, run_name=run_name)
        except Exception as exc:  # keep sweep going on individual-run failures
            print(f"  FAILED: {exc!r}")
            continue
        done += 1
        summary.append((name, accuracy))

    wall_minutes = (time.time() - sweep_start) / 60
    print(f"\nSweep complete: {done} ran, {skipped} skipped, {wall_minutes:.1f} min total")
    return summary


# ---------------------------------------------------------------------------
# Argparse + entry point
# ---------------------------------------------------------------------------


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Train, sweep, and summarize MNIST classification experiments.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=str, help="Path to a single RunConfig YAML")
    source.add_argument("--sweep", type=str, help="Path to a sweep YAML")
    source.add_argument("--summary", action="store_true", help="Print ranked summary of all runs under results/")

    parser.add_argument("--dry-run", action="store_true", help="Print the expanded sweep without running")

    # Ad-hoc sweep flags (used when neither --config nor --sweep is given)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY.keys()), help="Architectures to run")
    parser.add_argument("--augmentations", nargs="+", choices=list(AUGMENTATION_REGISTRY.keys()), help="Augmentation pipelines to run")
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZER_NAMES, help="Optimizers to run")
    parser.add_argument("--schedulers", nargs="+", choices=SCHEDULER_NAMES, help="Schedulers to run")

    # Single-knob overrides / ad-hoc defaults
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # Summary options
    parser.add_argument("--top", type=int, default=20, help="Top-N for --summary")
    parser.add_argument("--out", type=str, default=None, help="Write --summary table to this markdown file")

    # Paths
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    return parser


def main():
    parser = build_argument_parser()
    arguments = parser.parse_args()

    models_dir = Path(arguments.models_dir)
    results_dir = Path(arguments.results_dir)

    if arguments.summary:
        print_summary(results_dir, top=arguments.top, out=arguments.out)
        return

    device = get_device()

    if arguments.config:
        cfg = load_config(arguments.config)
        if arguments.epochs != 15:
            cfg.training.epochs = arguments.epochs
        run_single(cfg, device, models_dir=models_dir, results_dir=results_dir)
        return

    if arguments.sweep:
        with open(arguments.sweep, "r", encoding="utf-8") as f:
            sweep = yaml.safe_load(f) or {}
        sweep_dicts = expand_sweep(sweep)
        if arguments.epochs != 15:
            for cfg_dict in sweep_dicts:
                cfg_dict.setdefault("training", {})["epochs"] = arguments.epochs
        run_sweep(sweep_dicts, device, models_dir=models_dir, results_dir=results_dir, dry_run=arguments.dry_run)
        return

    # Ad-hoc sweep from CLI flags (CIFAR style). Requires at least one axis flag.
    axis_flags = [arguments.models, arguments.augmentations, arguments.optimizers, arguments.schedulers]
    if not any(axis_flags):
        parser.error("Nothing to do. Pass --config, --sweep, --summary, or any of --models/--augmentations/--optimizers/--schedulers.")

    arguments.models = arguments.models or ["mlp"]
    arguments.augmentations = arguments.augmentations or ["none"]
    arguments.optimizers = arguments.optimizers or ["adam"]
    arguments.schedulers = arguments.schedulers or ["cosine"]

    sweep_dicts = build_adhoc_sweep(arguments)
    run_sweep(sweep_dicts, device, models_dir=models_dir, results_dir=results_dir, dry_run=arguments.dry_run)


if __name__ == "__main__":
    main()

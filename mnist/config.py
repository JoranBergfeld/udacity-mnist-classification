"""YAML-backed experiment configuration with a dataclass schema."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    batch_size: int = 128
    val_split: float = 0.1
    augmentation: str = "none"
    num_workers: int = 0


@dataclass
class ModelConfig:
    name: str = "mlp"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9
    nesterov: bool = False


@dataclass
class SchedulerConfig:
    name: str = "none"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    epochs: int = 10
    label_smoothing: float = 0.0
    early_stopping_patience: int = 0  # 0 disables


@dataclass
class RunConfig:
    name: str = "run"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        return cls(
            name=d.get("name", "run"),
            seed=d.get("seed", 42),
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            optimizer=OptimizerConfig(**d.get("optimizer", {})),
            scheduler=SchedulerConfig(**d.get("scheduler", {})),
            training=TrainingConfig(**d.get("training", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return RunConfig.from_dict(raw)


def dump_config(cfg: RunConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)

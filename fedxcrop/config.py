"""Configuration system for FedXCrop experiments.

A single YAML file per experiment is loaded into nested dataclasses. Every run
writes its resolved configuration, a hash of that configuration, the git commit
it ran from, library versions, and the device name into its run directory, so
any result file can be traced back to the exact code and settings that made it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, get_args, get_origin

import yaml


@dataclass
class DataConfig:
    """Where the images live and how they are turned into tensors."""

    root: str = "plantvillage dataset"
    variant: str = "color"
    segmented_variant: str = "segmented"
    image_size: int = 224
    splits_dir: str = "splits"
    split_seed: int = 42
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class ModelConfig:
    name: str = "mobilenet_v2"
    num_classes: int = 38
    pretrained: bool = True


@dataclass
class CentralizedConfig:
    """Phase 3 centralized baseline, trained from ImageNet initialization."""

    epochs: int = 20
    optimizer: str = "adam"
    lr: float = 1e-3
    lr_step_size: int = 7
    lr_gamma: float = 0.1


@dataclass
class PartitionConfig:
    """How the train split is divided across simulated clients."""

    scheme: str = "dirichlet"  # "iid" or "dirichlet"
    num_clients: int = 5
    alpha: float = 0.5
    min_samples_per_client: int = 10
    seed: int = 0


@dataclass
class FederatedConfig:
    """Phase 4 federated run.

    init is "imagenet" for the corrected protocol. The legacy reproduction sets
    it to "checkpoint" and points init_checkpoint at the centralized model, which
    is the flaw the corrected protocol removes.
    """

    strategy: str = "fedavg"  # "fedavg" or "fedprox"
    rounds: int = 30
    local_epochs: int = 1
    mu: float = 0.01
    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    fraction_fit: float = 1.0
    init: str = "imagenet"
    init_checkpoint: Optional[str] = None
    train_subset_size: Optional[int] = None


@dataclass
class XAIConfig:
    """Phase 5 attribution methods and their evaluation."""

    methods: list[str] = field(default_factory=lambda: ["gradcam", "smoothgrad"])
    images_per_class: int = 10
    smoothgrad_samples: int = 20
    smoothgrad_noise_fraction: float = 0.1
    occlusion_step: float = 0.02
    topk_fraction: float = 0.2
    bootstrap_resamples: int = 1000


@dataclass
class EvalConfig:
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 42
    confidence: float = 0.95


@dataclass
class ExperimentConfig:
    """Top level configuration, one per experiment YAML file."""

    name: str = "unnamed"
    seed: int = 0
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    runs_dir: str = "runs"
    results_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    centralized: CentralizedConfig = field(default_factory=CentralizedConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_dir) / self.name


def _coerce(value: Any, target_type: Any) -> Any:
    """Convert a YAML scalar into the type the dataclass field declares."""
    origin = get_origin(target_type)
    if origin is not None:
        args = [a for a in get_args(target_type) if a is not type(None)]
        if value is None:
            return None
        if origin is list:
            return [_coerce(v, args[0]) for v in value] if args else list(value)
        return _coerce(value, args[0]) if args else value
    if target_type in (int, float, str, bool) and value is not None:
        return target_type(value)
    return value


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Build a (possibly nested) dataclass from a plain dictionary."""
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        f = known[key]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[key] = _from_dict(f.type, value)
        elif isinstance(f.type, str):
            # Postponed annotations: resolve against this module's namespace.
            resolved = getattr(sys.modules[__name__], f.type, None)
            if is_dataclass(resolved) and isinstance(value, dict):
                kwargs[key] = _from_dict(resolved, value)
            else:
                kwargs[key] = _coerce(value, resolved if resolved else f.type)
        else:
            kwargs[key] = _coerce(value, f.type)
    return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dictionary."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _apply_dotted(data: dict, dotted: str, raw_value: str) -> dict:
    """Apply a single "a.b.c=value" command line override."""
    keys = dotted.split(".")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError:
        value = raw_value
    patch: Any = value
    for key in reversed(keys):
        patch = {key: patch}
    return _deep_merge(data, patch)


def load_config(path: str | Path, overrides: Optional[list[str]] = None) -> ExperimentConfig:
    """Load a YAML config, apply "key.sub=value" overrides, return the dataclass.

    A config may set `base: other.yaml` to inherit from another file in the same
    directory, so shared settings are written once.
    """
    path = Path(path)
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}

    base_name = data.pop("base", None)
    if base_name:
        base_data = yaml.safe_load(open(path.parent / base_name)) or {}
        base_data.pop("base", None)
        data = _deep_merge(base_data, data)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        dotted, raw_value = item.split("=", 1)
        data = _apply_dotted(data, dotted.strip(), raw_value.strip())

    return _from_dict(ExperimentConfig, data)


def to_dict(cfg: Any) -> Any:
    """Recursively convert a config dataclass into plain dictionaries."""
    return dataclasses.asdict(cfg) if is_dataclass(cfg) else cfg


def config_hash(cfg: ExperimentConfig) -> str:
    """Stable 12 character hash of the resolved config, ignoring output paths."""
    data = to_dict(cfg)
    for key in ("runs_dir", "results_dir"):
        data.pop(key, None)
    data.get("data", {}).pop("num_workers", None)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def git_commit() -> dict[str, Any]:
    """Current commit hash and whether the working tree has uncommitted changes."""
    def run(args: list[str]) -> Optional[str]:
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run(["git", "rev-parse", "HEAD"])
    dirty = run(["git", "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def library_versions() -> dict[str, str]:
    """Versions of the libraries whose behaviour can change a result."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "torchvision", "numpy", "scipy", "sklearn", "pandas", "flwr", "captum"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def device_name(device: str = "auto") -> str:
    """Resolve "auto" to the best available device and name the accelerator."""
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device == "cuda" and torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    if device == "mps":
        return f"mps:{platform.processor()}"
    return "cpu"


def resolve_device(device: str = "auto"):
    """Return the torch device to run on."""
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def save_run_metadata(run_dir: str | Path, cfg: ExperimentConfig) -> Path:
    """Write the provenance record for a run and return its path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as handle:
        yaml.safe_dump(to_dict(cfg), handle, sort_keys=False)

    metadata = {
        "name": cfg.name,
        "config_hash": config_hash(cfg),
        "git": git_commit(),
        "libraries": library_versions(),
        "device": device_name(cfg.device),
        "argv": sys.argv,
        "cwd": os.getcwd(),
    }
    meta_path = run_dir / "run_metadata.json"
    with open(meta_path, "w") as handle:
        json.dump(metadata, handle, indent=2)
    return meta_path

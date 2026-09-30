"""Experiment configuration: YAML on disk, frozen dataclasses in memory."""

import dataclasses
import math
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ARCHITECTURES = ("unet", "attention_unet")
LOSSES = ("dice", "dice_ce", "dice_focal")
SCHEDULERS = ("cosine",)
BLEND_MODES = ("gaussian", "constant")
DEVICES = ("cuda", "cpu")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _choice(value: str, options: tuple[str, ...], where: str) -> None:
    _require(value in options, f"{where} must be one of {options}, got {value!r}")


@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float
    test: float
    # separate from the training seed, so a seed ablation can never move the split
    seed: int

    def __post_init__(self) -> None:
        _require(self.seed >= 0, "data.split.seed must be non-negative")
        parts = (self.train, self.val, self.test)
        _require(
            all(0.0 < p < 1.0 for p in parts),
            f"split fractions must lie in (0, 1): {parts}",
        )
        _require(
            math.isclose(sum(parts), 1.0),
            f"split fractions must sum to 1, got {sum(parts)}",
        )


@dataclass(frozen=True)
class LinkConfig:
    thumbnail_factor: int
    threshold: float
    duplicate_threshold: float

    def __post_init__(self) -> None:
        _require(self.thumbnail_factor >= 1, "data.link.thumbnail_factor must be >= 1")
        _require(0.0 < self.threshold <= 1.0, "data.link.threshold must lie in (0, 1]")
        _require(
            self.threshold <= self.duplicate_threshold <= 1.0,
            "data.link.duplicate_threshold must lie in [threshold, 1]",
        )


@dataclass(frozen=True)
class DataConfig:
    root: Path
    processed_dir: Path
    splits_file: Path
    split: SplitConfig
    link: LinkConfig
    regions: dict[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        _require(bool(self.regions), "data.regions must define at least one region")
        for name, labels in self.regions.items():
            distinct = len(labels) > 0 and len(set(labels)) == len(labels)
            _require(
                distinct, f"data.regions.{name} needs distinct labels, got {labels}"
            )


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    in_channels: int
    out_channels: int
    channels: tuple[int, ...]
    strides: tuple[int, ...]
    num_res_units: int
    dropout: float

    def __post_init__(self) -> None:
        _choice(self.architecture, ARCHITECTURES, "model.architecture")
        _require(self.in_channels > 0, "model.in_channels must be positive")
        _require(
            self.out_channels > 1,
            "model.out_channels must include background and >= 1 class",
        )
        _require(all(c > 0 for c in self.channels), "model.channels must be positive")
        _require(
            len(self.strides) == len(self.channels) - 1,
            "model.strides needs one entry per downsampling step, len(channels) - 1",
        )
        _require(all(s > 0 for s in self.strides), "model.strides must be positive")
        _require(self.num_res_units >= 0, "model.num_res_units must be non-negative")
        _require(0.0 <= self.dropout < 1.0, "model.dropout must lie in [0, 1)")


@dataclass(frozen=True)
class TrainConfig:
    patch_size: tuple[int, int, int]
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    loss: str
    scheduler: str
    amp: bool
    foreground_crop_prob: float
    val_every: int

    def __post_init__(self) -> None:
        _require(
            all(s > 0 for s in self.patch_size), "train.patch_size must be positive"
        )
        _require(self.batch_size > 0, "train.batch_size must be positive")
        _require(self.epochs > 0, "train.epochs must be positive")
        _require(self.learning_rate > 0.0, "train.learning_rate must be positive")
        _require(self.weight_decay >= 0.0, "train.weight_decay must be non-negative")
        _choice(self.loss, LOSSES, "train.loss")
        _choice(self.scheduler, SCHEDULERS, "train.scheduler")
        _require(
            0.0 <= self.foreground_crop_prob <= 1.0,
            "train.foreground_crop_prob must lie in [0, 1]",
        )
        _require(self.val_every > 0, "train.val_every must be positive")


@dataclass(frozen=True)
class AugmentConfig:
    enabled: bool
    flip_prob: float
    intensity_prob: float
    intensity_scale: float
    intensity_shift: float

    def __post_init__(self) -> None:
        _require(0.0 <= self.flip_prob <= 1.0, "augment.flip_prob must lie in [0, 1]")
        _require(
            0.0 <= self.intensity_prob <= 1.0,
            "augment.intensity_prob must lie in [0, 1]",
        )
        _require(self.intensity_scale >= 0.0, "augment.intensity_scale must be >= 0")
        _require(self.intensity_shift >= 0.0, "augment.intensity_shift must be >= 0")


@dataclass(frozen=True)
class InferenceConfig:
    sw_batch_size: int
    overlap: float
    blend_mode: str
    # null scores this config's own checkpoint; a run name reuses that run's model, which
    # lets inference-only ablations evaluate a trained model without retraining it
    checkpoint_run: str | None

    def __post_init__(self) -> None:
        _require(self.sw_batch_size > 0, "inference.sw_batch_size must be positive")
        _require(0.0 <= self.overlap < 1.0, "inference.overlap must lie in [0, 1)")
        _choice(self.blend_mode, BLEND_MODES, "inference.blend_mode")


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    num_threads: int
    cache_on_device: bool
    output_dir: Path
    tracking_dir: Path
    experiment: str

    def __post_init__(self) -> None:
        _choice(self.device, DEVICES, "runtime.device")
        _require(self.num_threads > 0, "runtime.num_threads must be positive")
        _require(bool(self.experiment), "runtime.experiment must be non-empty")


@dataclass(frozen=True)
class Config:
    name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    augment: AugmentConfig
    inference: InferenceConfig
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        _require(bool(self.name), "name must be non-empty")
        _require(self.seed >= 0, "seed must be non-negative")
        # a remainder after strided downsampling misaligns the decoder skip connections
        factor = math.prod(self.model.strides)
        _require(
            all(s % factor == 0 for s in self.train.patch_size),
            f"train.patch_size {self.train.patch_size} must be divisible by {factor}",
        )
        used = {label for labels in self.data.regions.values() for label in labels}
        _require(
            used <= set(range(1, self.model.out_channels)),
            f"region labels {sorted(used)} must be foreground classes of "
            f"model.out_channels={self.model.out_channels}",
        )


def _coerce(value: Any, hint: Any, where: str) -> Any:
    if dataclasses.is_dataclass(hint):
        return _build(hint, value, where)
    if isinstance(hint, types.UnionType):
        options = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        _require(len(options) == 1, f"no config coercion for type {hint} at {where}")
        return None if value is None else _coerce(value, options[0], where)
    if hint is Path:
        _require(
            isinstance(value, str) and value != "",
            f"{where} must be a path, got {value!r}",
        )
        return Path(value)
    if hint is bool:
        _require(
            isinstance(value, bool), f"{where} must be true or false, got {value!r}"
        )
        return value
    if hint is int:
        is_int = isinstance(value, int) and not isinstance(value, bool)
        _require(is_int, f"{where} must be an integer, got {value!r}")
        return value
    if hint is float:
        # pyyaml follows yaml 1.1, where 1e-4 without a dot is a string, not a float
        is_number = isinstance(value, int | float) and not isinstance(value, bool)
        _require(is_number, f"{where} must be a number, got {value!r}")
        return float(value)
    if hint is str:
        _require(isinstance(value, str), f"{where} must be a string, got {value!r}")
        return value
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is tuple:
        _require(isinstance(value, list), f"{where} must be a list, got {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            args = (args[0],) * len(value)
        _require(
            len(value) == len(args),
            f"{where} must have {len(args)} entries, got {value!r}",
        )
        return tuple(
            _coerce(v, h, f"{where}[{i}]") for i, (v, h) in enumerate(zip(value, args))
        )
    if origin is dict:
        _require(isinstance(value, dict), f"{where} must be a mapping, got {value!r}")
        key_hint, value_hint = args
        return {
            _coerce(k, key_hint, where): _coerce(v, value_hint, f"{where}.{k}")
            for k, v in value.items()
        }
    raise TypeError(f"no config coercion for type {hint} at {where}")


def _build(cls: type, raw: Any, where: str) -> Any:
    _require(isinstance(raw, dict), f"{where} must be a mapping, got {raw!r}")
    hints = typing.get_type_hints(cls)
    names = [field.name for field in dataclasses.fields(cls)]
    unknown = sorted(set(raw) - set(names))
    missing = [name for name in names if name not in raw]
    _require(not unknown, f"unknown keys in {where}: {unknown}")
    _require(not missing, f"missing keys in {where}: {missing}")
    return cls(
        **{name: _coerce(raw[name], hints[name], f"{where}.{name}") for name in names}
    )


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    _require(path not in seen, f"config extends itself: {' -> '.join(map(str, seen))}")
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    _require(isinstance(raw, dict), f"{path} must hold a mapping")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    # an ablation states only what it changes; everything else comes from its parent
    return _merge(_read(path.parent / parent, (*seen, path)), raw)


def load_config(path: str | Path) -> Config:
    return _build(Config, _read(Path(path)), "config")


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def save_config(cfg: Config, path: str | Path) -> None:
    text = yaml.safe_dump(_plain(dataclasses.asdict(cfg)), sort_keys=False)
    Path(path).write_text(text, encoding="utf-8", newline="\n")

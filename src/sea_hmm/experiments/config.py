from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

import torch

from ..config import _check_keys, _mapping, _read_experiment, _read_yaml, _resolve_experiment


@dataclass(frozen=True)
class DataConfig:
    mode: str
    dataset_path: Path
    prefix_lengths: list[int]
    lookahead: int
    train_size: int
    validation_size: int
    test_size: int


@dataclass(frozen=True)
class ModelConfig:
    type: str
    width: int
    layers: int
    dropout: float
    heads: int = 1
    feedforward: int = 128


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    ensemble_size: int


@dataclass(frozen=True)
class EvaluationConfig:
    modes: list[str]
    continuation_method: str
    batch_size: int
    num_continuations: int = 64
    max_enumerated_suffixes: int = 256


@dataclass(frozen=True)
class RunConfig:
    intent: dict
    device: torch.device
    seed: int
    output_dir: Path
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    def dataset_metadata(self) -> dict:
        data = asdict(self.data)
        del data["mode"], data["dataset_path"]
        return {"version": 1, "intent": self.intent, "seed": self.seed, "data": data}

    def snapshot(self) -> dict:
        snapshot = asdict(self)
        snapshot["device"] = str(self.device)
        snapshot["output_dir"] = str(self.output_dir)
        snapshot["data"]["dataset_path"] = str(self.data.dataset_path)
        return snapshot


def _count(value: object, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _path(base: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a path string")
    return (base / value).resolve()


def load_run_config(path: Path) -> RunConfig:
    path = path.resolve()
    raw = _mapping(_read_yaml(path), "run config")
    _check_keys(
        raw, {"intent", "seed", "output_dir", "data", "model", "training", "evaluation"}, {"device"}, "run config"
    )
    _count(raw["seed"], "seed", 0)
    definition = raw["intent"]
    intent = (
        _resolve_experiment(definition, path.parent) if isinstance(definition, dict)
        else _read_experiment(_path(path.parent, definition, "intent"))
    )
    device_name = raw.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"experiment device {device} is unavailable")

    data = _mapping(raw["data"], "data")
    counts = ("lookahead", "train_size", "validation_size", "test_size")
    _check_keys(data, {"mode", "dataset_path", "prefix_lengths", *counts}, set(), "data")
    if data["mode"] not in ("pre_generated", "online"):
        raise ValueError("data.mode must be pre_generated or online")
    for name in counts:
        _count(data[name], f"data.{name}")
    prefix_lengths = data["prefix_lengths"]
    if not isinstance(prefix_lengths, list) or not prefix_lengths:
        raise ValueError("data.prefix_lengths must be a nonempty list of positive integers")
    for length in prefix_lengths:
        _count(length, "prefix length")
    anchor = intent.get("anchor", "prefix_end")
    if intent["type"] == "anchored" and type(anchor) is int and anchor >= min(prefix_lengths):
        raise ValueError("anchor must be inside every configured prefix")
    data_config = DataConfig(**{**data, "dataset_path": _path(path.parent, data["dataset_path"], "dataset_path")})

    model = _mapping(raw["model"], "model")
    _check_keys(model, {"type", "width", "layers", "dropout"}, {"heads", "feedforward"}, "model")
    model_config = ModelConfig(**model)
    if model_config.type not in ("transformer", "lstm"):
        raise ValueError("model.type must be transformer or lstm")
    for name in ("width", "layers", "heads", "feedforward"):
        _count(getattr(model_config, name), f"model.{name}")
    if type(model_config.dropout) not in (int, float) or not 0 <= model_config.dropout < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    if model_config.type == "transformer" and model_config.width % model_config.heads:
        raise ValueError("transformer width must be divisible by its number of heads")

    training = _mapping(raw["training"], "training")
    _check_keys(training, {"epochs", "batch_size", "learning_rate", "ensemble_size"}, set(), "training")
    for name in ("epochs", "batch_size", "ensemble_size"):
        _count(training[name], f"training.{name}")
    learning_rate = training["learning_rate"]
    if type(learning_rate) not in (int, float) or not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("training.learning_rate must be finite and positive")

    evaluation = _mapping(raw["evaluation"], "evaluation")
    _check_keys(
        evaluation, {"modes", "continuation_method", "batch_size"},
        {"num_continuations", "max_enumerated_suffixes"}, "evaluation"
    )
    evaluation_config = EvaluationConfig(**evaluation)
    modes = evaluation_config.modes
    if not isinstance(modes, list) or not modes or any(mode not in ("oracle", "ensemble", "single") for mode in modes):
        raise ValueError("evaluation.modes must select from oracle, ensemble and single")
    if len(set(modes)) != len(modes):
        raise ValueError("evaluation.modes must be unique")
    if "ensemble" in modes and training["ensemble_size"] < 2:
        raise ValueError("ensemble evaluation requires at least two trained members")
    if evaluation_config.continuation_method not in ("sample", "enumerate"):
        raise ValueError("evaluation.continuation_method must be sample or enumerate")
    for name in ("batch_size", "num_continuations", "max_enumerated_suffixes"):
        _count(getattr(evaluation_config, name), f"evaluation.{name}")

    return RunConfig(
        intent, device, raw["seed"], _path(path.parent, raw["output_dir"], "output_dir"),
        data_config, model_config, TrainingConfig(**training), evaluation_config
    )

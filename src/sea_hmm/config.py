"""Load HMM and latent-intent configurations from YAML."""

from pathlib import Path

import torch
import yaml

from .data import AnchoredIntent, StaticIntent
from .hmm import CpGIslandModel, GaussianRegimeSwitchingHMM, Mess3, StateEmittingHMM


_HMM_TYPES = {
    "mess3": Mess3,
    "cpg_island": CpGIslandModel,
    "gaussian_regime_switching": GaussianRegimeSwitchingHMM,
}


def _read_yaml(path: Path) -> object:
    with path.open(encoding="utf-8") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {path}: {exc}") from exc


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return value


def _check_keys(config: dict, required: set[str], optional: set[str], name: str) -> None:
    missing = required - config.keys()
    unexpected = config.keys() - required - optional
    if missing or unexpected:
        raise ValueError(f"{name} has missing keys {sorted(missing)} and unexpected keys {sorted(unexpected)}")


def _build_hmm(config: object, device, dtype) -> StateEmittingHMM:
    config = _mapping(config, "HMM configuration")
    _check_keys(config, {"type"}, {"params"}, "HMM configuration")
    hmm_type = config["type"]
    if not isinstance(hmm_type, str) or hmm_type not in _HMM_TYPES:
        raise ValueError(f"unknown HMM type {hmm_type!r}")
    params = _mapping(config.get("params", {}), "HMM params")
    if "device" in params or "dtype" in params:
        raise ValueError("pass device and dtype to the YAML loader")
    try:
        return _HMM_TYPES[hmm_type](**params, device=device, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {hmm_type} params: {exc}") from exc


def load_hmm(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> StateEmittingHMM:
    """Load a process YAML file containing a type and optional params."""
    return _build_hmm(_read_yaml(Path(path)), device, dtype)


def load_experiment(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> AnchoredIntent | StaticIntent:
    """Load a latent-intent experiment with inline or file-based HMMs."""
    path = Path(path)
    config = _mapping(_read_yaml(path), "experiment configuration")

    def build_component(specification: object) -> StateEmittingHMM:
        if isinstance(specification, str):
            return load_hmm(path.parent / specification, device=device, dtype=dtype)
        return _build_hmm(specification, device, dtype)

    experiment_type = config.get("type")
    if experiment_type == "anchored":
        _check_keys(config, {"type", "hmm", "target_probs"}, {"anchor"}, "anchored configuration")
        return AnchoredIntent(
            build_component(config["hmm"]), config["target_probs"], anchor=config.get("anchor", "prefix_end")
        )
    if experiment_type == "static":
        _check_keys(config, {"type", "components", "prior", "target_probs"}, set(), "static configuration")
        if not isinstance(config["components"], list):
            raise ValueError("components must be a YAML sequence of HMM configurations or paths")
        components = [build_component(component) for component in config["components"]]
        return StaticIntent(components, config["prior"], config["target_probs"])
    raise ValueError(f"unknown experiment type {experiment_type!r}")

"""Load HMM and latent-intent configurations from YAML."""

from pathlib import Path

import torch
import yaml

from .data import AnchoredIntent, StaticIntent
from .hmm import StateEmittingHMM


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
    _check_keys(
        config,
        {"type", "num_states", "transition_probs", "emissions"},
        {"name", "initial_probs"},
        "HMM configuration",
    )
    hmm_type = config["type"]
    if hmm_type not in ("discrete", "continuous"):
        raise ValueError(f"unknown HMM type {hmm_type!r}; expected discrete or continuous")
    num_states = config["num_states"]
    if type(num_states) is not int or num_states < 1:
        raise ValueError("num_states must be a positive integer")
    transition = _matrix(config["transition_probs"], "transition_probs")
    if transition.shape != (num_states, num_states):
        raise ValueError(f"transition_probs must have shape ({num_states}, {num_states})")
    emissions = _matrix(config["emissions"], "emissions")
    if emissions.shape[0] != num_states:
        raise ValueError(f"emissions must have {num_states} rows")
    if hmm_type == "discrete":
        emission_params = {"categorical": emissions}
    else:
        if emissions.shape[1] != 2:
            raise ValueError("continuous emissions require one [mean, standard_deviation] row per state")
        emission_params = {"gaussian": (emissions[:, 0], emissions[:, 1])}

    return StateEmittingHMM(
        transition,
        name=config.get("name"),
        initial_probs=config.get("initial_probs"),
        device=device,
        dtype=dtype,
        **emission_params,
    )


def _matrix(values: object, name: str) -> torch.Tensor:
    try:
        matrix = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric matrix") from exc
    if matrix.ndim != 2 or not all(matrix.shape):
        raise ValueError(f"{name} must be a nonempty matrix")
    return matrix


def load_hmm(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> StateEmittingHMM:
    """Load a discrete or continuous HMM from a YAML state and emission specification."""
    return _build_hmm(_read_yaml(Path(path)), device, dtype)


def load_experiment(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> AnchoredIntent | StaticIntent:
    """Load a latent-intent experiment with inline or file-based HMMs."""
    return _build_experiment(_read_experiment(Path(path)), device=device, dtype=dtype)


def _read_experiment(path: Path) -> dict:
    return _resolve_experiment(_read_yaml(path), path.parent)


def _resolve_experiment(config: object, base: Path) -> dict:
    config = _mapping(config, "experiment configuration")

    def resolve_component(specification: object) -> object:
        return _read_yaml(base / specification) if isinstance(specification, str) else specification

    experiment_type = config.get("type")
    if experiment_type == "anchored":
        _check_keys(config, {"type", "hmm", "target_probs"}, {"anchor"}, "anchored configuration")
        return {**config, "hmm": resolve_component(config["hmm"])}
    if experiment_type == "static":
        _check_keys(config, {"type", "components", "prior", "target_probs"}, set(), "static configuration")
        if not isinstance(config["components"], list):
            raise ValueError("components must be a YAML sequence of HMM configurations or paths")
        return {**config, "components": [resolve_component(component) for component in config["components"]]}
    raise ValueError(f"unknown experiment type {experiment_type!r}")


def _build_experiment(config: dict, *, device, dtype) -> AnchoredIntent | StaticIntent:
    if config["type"] == "anchored":
        return AnchoredIntent(
            _build_hmm(config["hmm"], device, dtype),
            config["target_probs"],
            anchor=config.get("anchor", "prefix_end"),
        )
    components = [_build_hmm(component, device, dtype) for component in config["components"]]
    return StaticIntent(components, config["prior"], config["target_probs"])

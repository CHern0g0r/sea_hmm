from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor


Lengths = int | Sequence[int] | Tensor


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    """Padded observations, sampled hidden paths, and observation lengths.

    For a row of length L, observations[:L] and hidden_states[:L]
    are valid. Observation padding is zero and hidden state padding is -1.
    """

    observations: Tensor
    hidden_states: Tensor
    lengths: Tensor


def _normalize_lengths(lengths: Lengths) -> tuple[int, ...]:
    if isinstance(lengths, Tensor):
        if lengths.ndim > 1:
            raise ValueError("lengths must be an integer or a one-dimensional sequence")
        lengths = lengths.tolist()
    if isinstance(lengths, Integral):
        lengths = (lengths,)
    if not isinstance(lengths, Sequence) or any(
        not isinstance(length, Integral) or isinstance(length, bool) or length < 0
        for length in lengths
    ):
        raise ValueError("lengths must contain nonnegative integers")
    return tuple(int(length) for length in lengths)


def _prepare_observations(observations, lengths, device) -> tuple[Tensor, tuple[int, ...]]:
    observations = torch.as_tensor(observations, device=device)
    if observations.ndim == 1:
        observations = observations.unsqueeze(0)
    if observations.ndim != 2:
        raise ValueError("observations must have shape [length] or [batch, observation_width]")
    batch_size, width = observations.shape
    lengths = (width,) * batch_size if lengths is None else _normalize_lengths(lengths)
    if len(lengths) != batch_size or any(length > width for length in lengths):
        raise ValueError("lengths must give one valid observation length per batch row")
    return observations, lengths


def _validate_count(value: int, name: str, *, minimum: int = 0) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")

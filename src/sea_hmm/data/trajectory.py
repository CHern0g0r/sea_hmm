from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

from torch import Tensor


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    """Padded observations, sampled hidden paths, and observation lengths.

    For a row of length L, observations[:L] and hidden_states[:L + 1]
    are valid. Observation padding is zero and hidden state padding is -1.
    """

    observations: Tensor
    hidden_states: Tensor
    lengths: Tensor


def _normalize_lengths(lengths: int | Sequence[int] | Tensor) -> tuple[int, ...]:
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

from collections.abc import Sequence

import torch
from torch import Tensor

from .._probabilities import _normalize_probabilities
from ._process import StateEmittingHMM


class Mess3(StateEmittingHMM):
    """Three states with categorical emissions that favor the emitting state's ID."""

    def __init__(
        self,
        switch_prob: float = 0.05,
        match_prob: float = 0.85,
        *,
        initial_probs: Sequence[float] | Tensor | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if not 0 <= switch_prob <= 0.5:
            raise ValueError("switch_prob must be in [0, 0.5]")
        if not 0 <= match_prob <= 1:
            raise ValueError("match_prob must be in [0, 1]")

        # Shai et al., Transformers Represent Belief State Geometry, Appendix A.3.
        transition = torch.full((3, 3), switch_prob, dtype=torch.float64, device="cpu")
        transition.fill_diagonal_(1 - 2 * switch_prob)
        emissions = torch.full((3, 3), (1 - match_prob) / 2, dtype=torch.float64, device="cpu")
        emissions.fill_diagonal_(match_prob)
        super().__init__(
            transition, categorical=emissions, initial_probs=initial_probs, device=device, dtype=dtype
        )


class CpGIslandModel(StateEmittingHMM):
    """Eight states ordered by region (background, island), then base (A, C, G, T).

    A state emits the current base.
    """

    def __init__(
        self,
        background_to_island: float = 0.001,
        island_to_background: float = 0.01,
        *,
        initial_probs: Sequence[float] | Tensor | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        region_transitions = _normalize_probabilities(
            [
                [1 - background_to_island, background_to_island],
                [island_to_background, 1 - island_to_background],
            ],
            (2, 2),
            "region transitions",
        )
        # Durbin et al., Biological Sequence Analysis, chapter 3, p. 50.
        base_transitions = torch.tensor(
            [
                [
                    [0.300, 0.205, 0.285, 0.210],
                    [0.322, 0.298, 0.078, 0.302],
                    [0.248, 0.246, 0.298, 0.208],
                    [0.177, 0.239, 0.292, 0.292],
                ],
                [
                    [0.180, 0.274, 0.426, 0.120],
                    [0.171, 0.368, 0.274, 0.188],
                    [0.161, 0.339, 0.375, 0.125],
                    [0.079, 0.355, 0.384, 0.182],
                ],
            ],
            dtype=torch.float64,
            device="cpu",
        )
        base_transitions /= base_transitions.sum(dim=-1, keepdim=True)
        transition = (
            region_transitions[:, None, :, None] * base_transitions.permute(1, 0, 2)[None]
        ).reshape(8, 8)
        emissions = torch.eye(4, dtype=torch.float64, device="cpu").repeat(2, 1)
        super().__init__(
            transition, categorical=emissions, initial_probs=initial_probs, device=device, dtype=dtype
        )


class GaussianRegimeSwitchingHMM(StateEmittingHMM):
    """An HMM with a scalar Gaussian emission from the current regime."""

    def __init__(
        self,
        transition_probs: Sequence[Sequence[float]] | Tensor = ((0.95, 0.05), (0.05, 0.95)),
        means: Sequence[float] | Tensor = (-1, 1),
        stds: Sequence[float] | Tensor = (1, 1),
        *,
        initial_probs: Sequence[float] | Tensor | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(
            transition_probs, gaussian=(means, stds), initial_probs=initial_probs, device=device, dtype=dtype
        )

from collections import defaultdict
from collections.abc import Sequence

import torch
from pomegranate.distributions import Categorical, Normal
from pomegranate.hmm import DenseHMM
from torch import Tensor

from .._probabilities import _normalize_probabilities
from ..data.trajectory import (
    Lengths,
    TrajectoryBatch,
    _normalize_lengths,
    _prepare_observations,
    _validate_count,
)


def _stationary_distribution(transition: Tensor) -> Tensor:
    size = transition.shape[0]
    equations = transition.T - torch.eye(size, dtype=transition.dtype, device=transition.device)
    equations[-1] = 1
    if torch.linalg.matrix_rank(equations) < size:
        raise ValueError("transitions require explicit initial_probs without a unique stationary distribution")
    target = transition.new_zeros(size)
    target[-1] = 1
    stationary = torch.linalg.solve(equations, target).clamp_min(0)
    return stationary / stationary.sum()


class StateEmittingHMM:
    """A fixed-parameter HMM with one emitting state per observation."""

    def __init__(
        self,
        transition_probs: Sequence[Sequence[float]] | Tensor,
        *,
        initial_probs: Sequence[float] | Tensor | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        categorical: Sequence[Sequence[float]] | Tensor | None = None,
        gaussian: tuple[Sequence[float] | Tensor, Sequence[float] | Tensor] | None = None,
    ) -> None:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if (categorical is None) == (gaussian is None):
            raise ValueError("provide exactly one of categorical probabilities or Gaussian parameters")
        transition = torch.as_tensor(transition_probs, dtype=torch.float64, device="cpu")
        if transition.ndim != 2 or not transition.shape[0]:
            raise ValueError("transition_probs must be a nonempty square matrix")
        num_states = transition.shape[0]
        transition = _normalize_probabilities(transition, (num_states, num_states), "transition_probs")
        initial = (
            _stationary_distribution(transition)
            if initial_probs is None
            else _normalize_probabilities(initial_probs, (num_states,), "initial_probs")
        )

        transition = transition.to(dtype=dtype)
        initial = initial.to(dtype=dtype)
        self._categorical = None
        if categorical is not None:
            categorical = torch.as_tensor(categorical, dtype=torch.float64, device="cpu")
            if categorical.ndim != 2 or not categorical.shape[1]:
                raise ValueError("categorical probabilities must have a nonempty symbol axis")
            categorical = _normalize_probabilities(
                categorical, (num_states, categorical.shape[1]), "categorical probabilities"
            ).to(dtype=dtype)
            distributions = [Categorical(probs[None], frozen=True) for probs in categorical]
            self._categorical = categorical.to(device=device)
            self._log_categorical = self._categorical.log()
        else:
            assert gaussian is not None
            means, stds = (
                torch.as_tensor(values, dtype=dtype, device="cpu").detach().clone()
                for values in gaussian
            )
            variances = stds.square()
            if means.shape != (num_states,) or stds.shape != means.shape:
                raise ValueError("provide one Gaussian mean and standard deviation per state")
            if not torch.isfinite(means).all() or not (
                (stds > 0) & torch.isfinite(variances) & (variances > 0)
            ).all():
                raise ValueError(
                    "Gaussian parameters must have finite means and positive finite variances in dtype"
                )
            distributions = [
                Normal(mean[None], variance[None], covariance_type="diag", frozen=True)
                for mean, variance in zip(means, variances)
            ]
            self._means = means.to(device=device)
            self._stds = stds.to(device=device)

        self._hmm = DenseHMM(
            distributions=distributions,
            edges=transition,
            starts=initial,
            # The requested length is fixed independently of the hidden path.
            ends=torch.ones_like(initial),
            frozen=True,
        ).to(device=device, dtype=dtype)
        self._transition = transition.to(device=device)
        self._initial = initial.to(device=device)

    @property
    def device(self) -> torch.device:
        return self._transition.device

    @property
    def dtype(self) -> torch.dtype:
        return self._transition.dtype

    @property
    def num_states(self) -> int:
        return self._initial.numel()

    @property
    def num_symbols(self) -> int | None:
        """Number of categorical symbols, or None for continuous observations."""
        return None if self._categorical is None else self._categorical.shape[1]

    @property
    def observation_dtype(self) -> torch.dtype:
        return self.dtype if self.num_symbols is None else torch.long

    @torch.no_grad()
    def sample(
        self,
        lengths: Lengths,
        *,
        generator: torch.Generator | None = None,
    ) -> TrajectoryBatch:
        """Sample observations and their exact hidden states using the model device."""
        lengths = _normalize_lengths(lengths)
        lengths_tensor = torch.tensor(lengths, device=self.device, dtype=torch.long)
        width = max(lengths, default=0)
        observations, states = self._sample_paths(self._initial.expand(len(lengths), -1), width, generator)
        padding = torch.arange(width, device=self.device) >= lengths_tensor[:, None]
        observations.masked_fill_(padding, 0)
        states.masked_fill_(padding, -1)
        return TrajectoryBatch(observations, states, lengths_tensor)

    def _sample_paths(self, first_state_probs: Tensor, width: int, generator) -> tuple[Tensor, Tensor]:
        shape = (first_state_probs.shape[0], width)
        states = torch.empty(shape, device=self.device, dtype=torch.long)
        if states.numel() == 0:
            return states.to(dtype=self.observation_dtype), states

        states[:, 0] = torch.multinomial(first_state_probs, 1, generator=generator).squeeze(-1)
        for i in range(1, width):
            next_probs = self._transition[states[:, i - 1]]
            states[:, i] = torch.multinomial(next_probs, 1, generator=generator).squeeze(-1)

        if self._categorical is not None:
            emission_probs = self._categorical[states.flatten()]
            observations = torch.multinomial(emission_probs, 1, generator=generator).reshape(shape)
        else:
            noise = torch.randn(shape, device=self.device, dtype=self.dtype, generator=generator)
            observations = self._means[states] + self._stds[states] * noise
        return observations, states

    @torch.no_grad()
    def filter(self, observations: Tensor, *, lengths: Lengths | None = None) -> Tensor:
        """Return the posterior of each emitting state given observations through that state."""
        return self._track(observations, lengths, smoothing=False)

    @torch.no_grad()
    def smooth(self, observations: Tensor, *, lengths: Lengths | None = None) -> Tensor:
        """Return the posterior of each emitting state given the full valid sequence."""
        return self._track(observations, lengths, smoothing=True)

    def _track(
        self, observations: Tensor, lengths: Lengths | None, *, smoothing: bool
    ) -> Tensor:
        observations, lengths = _prepare_observations(observations, lengths, self.device)
        result = self._initial.new_zeros((*observations.shape, self.num_states))
        for indices, emissions, forward, evidence in self._forward_groups(observations, lengths):
            if torch.isneginf(evidence).any():
                raise ValueError("state posteriors require a nonzero sequence likelihood")
            if smoothing:
                forward = forward + self._hmm.backward(emissions=emissions)
            result[indices, :forward.shape[1]] = torch.softmax(forward, dim=-1)
        return result

    def _forward_groups(self, observations: Tensor, lengths: tuple[int, ...]):
        groups = defaultdict(list)
        for row, length in enumerate(lengths):
            if length:
                groups[length].append(row)

        for length, rows in groups.items():
            indices = torch.tensor(rows, device=self.device)
            valid = observations[:, :length].index_select(0, indices)
            emissions = self._log_emissions(valid)
            forward = self._hmm.forward(emissions=emissions)
            # Pomegranate can produce NaNs after an all-impossible prefix.
            impossible = torch.isneginf(forward).all(dim=-1).any(dim=-1)
            evidence = torch.logsumexp(forward[:, -1], dim=-1).masked_fill(impossible, -torch.inf)
            if not (torch.isfinite(evidence) | torch.isneginf(evidence)).all():
                raise ValueError("non-finite sequence log likelihood")
            yield indices, emissions, forward, evidence

    @torch.no_grad()
    def log_likelihood(self, observations: Tensor, *, lengths: Lengths | None = None) -> Tensor:
        """Return one sequence log likelihood per row, including zero for an empty row."""
        observations, lengths = _prepare_observations(observations, lengths, self.device)
        result = self._initial.new_zeros(len(lengths))
        for indices, _, _, evidence in self._forward_groups(observations, lengths):
            result[indices] = evidence
        return result

    def _prefix_summary(self, observations: Tensor, lengths: tuple[int, ...]) -> tuple[Tensor, Tensor]:
        evidence = self._initial.new_zeros(len(lengths))
        next_probs = self._initial.expand(len(lengths), -1).clone()
        for indices, _, forward, log_likelihood in self._forward_groups(observations, lengths):
            evidence[indices] = log_likelihood
            impossible = torch.isneginf(log_likelihood)[:, None]
            last = forward[:, -1].masked_fill(impossible, 0)
            predicted = torch.softmax(last, dim=-1) @ self._transition
            next_probs[indices] = predicted.masked_fill(impossible, 0)
        return evidence, next_probs

    @torch.no_grad()
    def sample_continuations(
        self,
        observations: Tensor,
        k: int,
        *,
        lengths: Lengths | None = None,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample suffixes conditional on each prefix, with shape [batch, num_samples, k]."""
        _validate_count(k, "k")
        _validate_count(num_samples, "num_samples", minimum=1)
        observations, lengths = _prepare_observations(observations, lengths, self.device)
        evidence, next_probs = self._prefix_summary(observations, lengths)
        if torch.isneginf(evidence).any():
            raise ValueError("continuations require a nonzero prefix likelihood")
        starts = next_probs.repeat_interleave(num_samples, dim=0)
        suffixes, _ = self._sample_paths(starts, k, generator)
        return suffixes.reshape(len(lengths), num_samples, k)

    def _log_emissions(self, observations: Tensor) -> Tensor:
        batch_size, width = observations.shape
        if observations.is_complex() or observations.dtype == torch.bool:
            raise ValueError("observations must be real numbers")
        if self._categorical is not None:
            num_symbols = self._categorical.shape[1]
            if not ((observations >= 0) & (observations < num_symbols)).all() or (
                observations.is_floating_point() and not (observations == observations.trunc()).all()
            ):
                raise ValueError(f"categorical observations must be integer IDs in [0, {num_symbols})")
            return self._log_categorical.T[observations.long()]

        values = observations.to(dtype=self.dtype).reshape(-1, 1)
        if not torch.isfinite(values).all():
            raise ValueError("Gaussian observations must be finite in dtype")
        return torch.stack(
            [distribution.log_probability(values) for distribution in self._hmm.distributions], dim=-1
        ).reshape(batch_size, width, self.num_states)

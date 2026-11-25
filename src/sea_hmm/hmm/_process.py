from collections import defaultdict
from collections.abc import Sequence

import torch
from pomegranate.distributions import Categorical, Normal
from pomegranate.hmm import DenseHMM
from torch import Tensor

from ..data import TrajectoryBatch
from ..data.trajectory import _normalize_lengths


def _normalize_probabilities(
    values: Sequence[float] | Sequence[Sequence[float]] | Tensor,
    shape: tuple[int, ...],
    name: str,
) -> Tensor:
    probabilities = torch.as_tensor(values, dtype=torch.float64, device="cpu").detach()
    if probabilities.shape != shape or not torch.isfinite(probabilities).all():
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    totals = probabilities.sum(dim=-1, keepdim=True)
    if (probabilities < 0).any() or not torch.allclose(
        totals, torch.ones_like(totals), rtol=0, atol=1e-6
    ):
        raise ValueError(f"{name} must contain nonnegative probabilities that sum to one")
    return probabilities / totals


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


class _Process:
    def __init__(
        self,
        transition: Tensor,
        *,
        initial_probs: Sequence[float] | Tensor | None,
        device: str | torch.device | None,
        dtype: torch.dtype,
        categorical: Tensor | None = None,
        gaussian: tuple[Tensor, Tensor] | None = None,
    ) -> None:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        num_states = transition.shape[0]
        initial = (
            _stationary_distribution(transition)
            if initial_probs is None
            else _normalize_probabilities(initial_probs, (num_states,), "initial_probs")
        )

        transition = transition.to(dtype=dtype)
        initial = initial.to(dtype=dtype)
        self._categorical = None
        if categorical is not None:
            categorical = categorical.to(dtype=dtype)
            distributions = [Categorical(probs[None], frozen=True) for probs in categorical]
            self._categorical = categorical.to(device=device)
            self._log_categorical = self._categorical.log()
        else:
            assert gaussian is not None
            means, stds = (parameters.to(dtype=dtype) for parameters in gaussian)
            variances = stds.square()
            if not torch.isfinite(means).all() or not (
                torch.isfinite(variances) & (variances > 0)
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

    @torch.no_grad()
    def sample(
        self,
        lengths: int | Sequence[int] | Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TrajectoryBatch:
        """Sample observations and their exact hidden states using the model device."""
        lengths = _normalize_lengths(lengths)
        batch_size, width = len(lengths), max(lengths, default=0)
        lengths_tensor = torch.tensor(lengths, device=self.device, dtype=torch.long)
        states = torch.empty((batch_size, width + 1), device=self.device, dtype=torch.long)
        observation_dtype = torch.long if self._categorical is not None else self.dtype
        if not batch_size:
            observations = torch.empty((0, 0), device=self.device, dtype=observation_dtype)
            return TrajectoryBatch(observations, states, lengths_tensor)

        states[:, 0] = torch.multinomial(self._initial, batch_size, True, generator=generator)
        for t in range(width):
            next_probs = self._transition[states[:, t]]
            states[:, t + 1] = torch.multinomial(next_probs, 1, generator=generator).squeeze(-1)

        if width == 0:
            observations = torch.empty((batch_size, 0), device=self.device, dtype=observation_dtype)
        elif self._categorical is not None:
            emission_probs = self._categorical[states[:, 1:].reshape(-1)]
            observations = torch.multinomial(emission_probs, 1, generator=generator)
            observations = observations.reshape(batch_size, width)
        else:
            emitting_states = states[:, :-1]
            noise = torch.randn((batch_size, width), device=self.device, dtype=self.dtype, generator=generator)
            observations = self._means[emitting_states] + self._stds[emitting_states] * noise

        positions = torch.arange(width + 1, device=self.device)
        observations.masked_fill_(positions[:-1] >= lengths_tensor[:, None], 0)
        states.masked_fill_(positions > lengths_tensor[:, None], -1)
        return TrajectoryBatch(observations, states, lengths_tensor)

    @torch.no_grad()
    def filter(self, observations: Tensor, *, lengths: Sequence[int] | Tensor | None = None) -> Tensor:
        """Return P(Z_t | Y[:t]) at every valid prefix, including the empty prefix."""
        return self._track(observations, lengths, smoothing=False)

    @torch.no_grad()
    def smooth(self, observations: Tensor, *, lengths: Sequence[int] | Tensor | None = None) -> Tensor:
        """Return P(Z_t | Y[:L]) at every valid state position in each length-L row."""
        return self._track(observations, lengths, smoothing=True)

    def _track(
        self, observations: Tensor, lengths: Sequence[int] | Tensor | None, *, smoothing: bool
    ) -> Tensor:
        observations = torch.as_tensor(observations, device=self.device)
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        if observations.ndim != 2:
            raise ValueError("observations must have shape [length] or [batch, observation_width]")
        batch_size, width = observations.shape
        lengths = (width,) * batch_size if lengths is None else _normalize_lengths(lengths)
        if len(lengths) != batch_size or any(length > width for length in lengths):
            raise ValueError("lengths must give one valid observation length per batch row")

        groups = defaultdict(list)
        for row, length in enumerate(lengths):
            groups[length].append(row)

        result = self._initial.new_zeros((batch_size, width + 1, self.num_states))
        for length, rows in groups.items():
            indices = torch.tensor(rows, device=self.device)
            if length == 0:
                result[indices, 0] = self._initial
                continue

            valid = observations[:, :length].index_select(0, indices)
            emissions = self._log_emissions(valid)
            forward = self._hmm.forward(emissions=emissions)
            if not torch.isfinite(torch.logsumexp(forward[:, -1], dim=-1)).all():
                raise ValueError("an observation sequence has zero or non-finite likelihood")

            if smoothing:
                backward = self._hmm.backward(emissions=emissions)
                beliefs = torch.softmax(forward + backward, dim=-1)
            elif self._categorical is not None:
                beliefs = torch.softmax(forward, dim=-1)
            else:
                # Gaussian forward messages at Z_t already include Y_t.
                predicted = torch.softmax(forward[:, :-1], dim=-1) @ self._transition
                prior = self._initial.expand(len(rows), 1, self.num_states)
                beliefs = torch.cat((prior, predicted), dim=1)
            result[indices, :length + 1] = beliefs
        return result

    def _log_emissions(self, observations: Tensor) -> Tensor:
        batch_size, width = observations.shape
        neutral = self._initial.new_zeros((batch_size, 1, self.num_states))
        if observations.is_complex() or observations.dtype == torch.bool:
            raise ValueError("observations must be real numbers")
        if self._categorical is not None:
            num_symbols = self._categorical.shape[1]
            if not ((observations >= 0) & (observations < num_symbols)).all() or (
                observations.is_floating_point() and not (observations == observations.trunc()).all()
            ):
                raise ValueError(f"categorical observations must be integer IDs in [0, {num_symbols})")
            log_probs = self._log_categorical.T[observations.long()]
            return torch.cat((neutral, log_probs), dim=1)

        values = observations.to(dtype=self.dtype).reshape(-1, 1)
        if not torch.isfinite(values).all():
            raise ValueError("Gaussian observations must be finite in dtype")
        log_probs = torch.stack(
            [distribution.log_probability(values) for distribution in self._hmm.distributions], dim=-1
        ).reshape(batch_size, width, self.num_states)
        return torch.cat((log_probs, neutral), dim=1)

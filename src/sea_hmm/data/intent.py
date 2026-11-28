from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .._probabilities import _normalize_probabilities
from .trajectory import (
    Lengths,
    TrajectoryBatch,
    _normalize_lengths,
    _prepare_observations,
    _validate_count,
)

if TYPE_CHECKING:
    from ..hmm import StateEmittingHMM


@dataclass(frozen=True, slots=True)
class IntentBatch:
    """Paired prefixes and extended trajectories with one intent and target per row.

    Use prefix_lengths to take the short view and trajectory.lengths for the long
    view. Static-mixture hidden state IDs are local to each row's component.
    """

    trajectory: TrajectoryBatch
    prefix_lengths: Tensor
    latent_ids: Tensor
    targets: Tensor
    anchors: Tensor | None = None


def _target_table(values, num_intents: int, model: StateEmittingHMM) -> Tensor:
    probabilities = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if probabilities.ndim != 2 or not probabilities.shape[1]:
        raise ValueError("target_probs must give every intent a distribution over target IDs")
    return _normalize_probabilities(
        probabilities, (num_intents, probabilities.shape[1]), "target_probs"
    ).to(device=model.device, dtype=model.dtype)


def _draw_targets(target_probs: Tensor, latent_ids: Tensor, generator: torch.Generator | None) -> Tensor:
    return torch.multinomial(target_probs[latent_ids], 1, generator=generator).squeeze(-1)


class AnchoredIntent:
    """An external target governed by the emitting state at a fixed anchor."""

    def __init__(
        self,
        hmm: StateEmittingHMM,
        target_probs: Sequence[Sequence[float]] | Tensor,
        *,
        anchor: int | str = "prefix_end",
    ) -> None:
        if anchor != "prefix_end":
            _validate_count(anchor, "anchor")
        self.hmm = hmm
        self.anchor = anchor
        self._target_probs = _target_table(target_probs, hmm.num_states, hmm)

    @property
    def target_probs(self) -> Tensor:
        """Return the conditional target distributions in anchor-state order."""
        return self._target_probs.clone()

    @torch.no_grad()
    def sample(
        self, prefix_lengths: Lengths, k: int, *, generator: torch.Generator | None = None
    ) -> IntentBatch:
        _validate_count(k, "k", minimum=1)
        lengths = _normalize_lengths(prefix_lengths)
        anchors = tuple(length - 1 if self.anchor == "prefix_end" else self.anchor for length in lengths)
        if any(not 0 <= anchor < length for anchor, length in zip(anchors, lengths)):
            raise ValueError("each anchor must be within its unclarified prefix")
        trajectory = self.hmm.sample(tuple(length + k for length in lengths), generator=generator)
        prefix_lengths = trajectory.lengths - k
        anchor_indices = torch.tensor(anchors, device=self.hmm.device, dtype=torch.long)
        rows = torch.arange(len(lengths), device=self.hmm.device)
        latent_ids = trajectory.hidden_states[rows, anchor_indices]
        targets = _draw_targets(self._target_probs, latent_ids, generator)
        return IntentBatch(trajectory, prefix_lengths, latent_ids, targets, anchor_indices)

    @torch.no_grad()
    def intent_posterior(
        self, observations: Tensor, *, anchors: Lengths, lengths: Lengths | None = None
    ) -> Tensor:
        observations, lengths = _prepare_observations(observations, lengths, self.hmm.device)
        single_anchor = isinstance(anchors, Integral) or (isinstance(anchors, Tensor) and anchors.ndim == 0)
        anchors = _normalize_lengths(anchors)
        if single_anchor:
            anchors = anchors * len(lengths)
        if len(anchors) != len(lengths) or any(anchor >= length for anchor, length in zip(anchors, lengths)):
            raise ValueError("anchors must give a valid emitting-state position for each row")
        anchor_indices = torch.tensor(anchors, device=self.hmm.device, dtype=torch.long)
        rows = torch.arange(len(lengths), device=self.hmm.device)
        return self.hmm.smooth(observations, lengths=lengths)[rows, anchor_indices]

    def predict_target(
        self, observations: Tensor, *, anchors: Lengths, lengths: Lengths | None = None
    ) -> Tensor:
        return self.intent_posterior(observations, anchors=anchors, lengths=lengths) @ self._target_probs

    def log_likelihood(self, observations: Tensor, *, lengths: Lengths | None = None) -> Tensor:
        return self.hmm.log_likelihood(observations, lengths=lengths)

    def sample_continuations(
        self,
        observations: Tensor,
        k: int,
        *,
        lengths: Lengths | None = None,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return self.hmm.sample_continuations(
            observations, k, lengths=lengths, num_samples=num_samples, generator=generator
        )


class StaticIntent:
    """An external target governed by the HMM selected for an entire episode."""

    def __init__(
        self,
        components: Sequence[StateEmittingHMM],
        prior: Sequence[float] | Tensor,
        target_probs: Sequence[Sequence[float]] | Tensor,
    ) -> None:
        self.components = tuple(components)
        if not self.components:
            raise ValueError("provide at least one HMM component")
        first = self.components[0]
        for component in self.components[1:]:
            if component.num_symbols != first.num_symbols:
                raise ValueError("components must share an observation space")
            if (component.device, component.dtype) != (first.device, first.dtype):
                raise ValueError("components must share a device and dtype")
        self.device = first.device
        self.observation_dtype = first.observation_dtype
        self._prior = _normalize_probabilities(prior, (len(self.components),), "prior").to(
            device=first.device, dtype=first.dtype
        )
        self._log_prior = self._prior.log()
        self._target_probs = _target_table(target_probs, len(self.components), first)

    @property
    def target_probs(self) -> Tensor:
        """Return the conditional target distributions in component order."""
        return self._target_probs.clone()

    @torch.no_grad()
    def sample(
        self, prefix_lengths: Lengths, k: int, *, generator: torch.Generator | None = None
    ) -> IntentBatch:
        _validate_count(k, "k", minimum=1)
        lengths = _normalize_lengths(prefix_lengths)
        shape = (len(lengths), max(lengths) + k if lengths else 0)
        observations = torch.zeros(shape, device=self.device, dtype=self.observation_dtype)
        states = torch.full(shape, -1, device=self.device, dtype=torch.long)
        prefix_lengths = torch.tensor(lengths, device=self.device, dtype=torch.long)
        total_lengths = prefix_lengths + k
        latent_ids = torch.multinomial(
            self._prior.expand(len(lengths), -1), 1, generator=generator
        ).squeeze(-1)
        for component_id, component in enumerate(self.components):
            rows = (latent_ids == component_id).nonzero(as_tuple=True)[0]
            if rows.numel():
                batch = component.sample(total_lengths[rows], generator=generator)
                width = batch.observations.shape[1]
                observations[rows, :width] = batch.observations
                states[rows, :width] = batch.hidden_states
        targets = _draw_targets(self._target_probs, latent_ids, generator)
        return IntentBatch(TrajectoryBatch(observations, states, total_lengths), prefix_lengths, latent_ids, targets)

    def _log_weights(self, observations: Tensor, lengths: Lengths | None) -> Tensor:
        observations, lengths = _prepare_observations(observations, lengths, self.device)
        likelihoods = [component.log_likelihood(observations, lengths=lengths) for component in self.components]
        return torch.stack(likelihoods, dim=-1) + self._log_prior

    def log_likelihood(self, observations: Tensor, *, lengths: Lengths | None = None) -> Tensor:
        return torch.logsumexp(self._log_weights(observations, lengths), dim=-1)

    def intent_posterior(self, observations: Tensor, *, lengths: Lengths | None = None) -> Tensor:
        return self._normalize_weights(self._log_weights(observations, lengths))

    @staticmethod
    def _normalize_weights(log_weights: Tensor) -> Tensor:
        if torch.isneginf(log_weights).all(dim=-1).any():
            raise ValueError("the prefix has zero likelihood under the mixture")
        return torch.softmax(log_weights, dim=-1)

    def predict_target(self, observations: Tensor, *, lengths: Lengths | None = None) -> Tensor:
        return self.intent_posterior(observations, lengths=lengths) @ self._target_probs

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
        _validate_count(k, "k")
        _validate_count(num_samples, "num_samples", minimum=1)
        observations, lengths = _prepare_observations(observations, lengths, self.device)
        summaries = [component._prefix_summary(observations, lengths) for component in self.components]
        log_weights = torch.stack([evidence for evidence, _ in summaries], dim=-1) + self._log_prior
        component_probs = self._normalize_weights(log_weights)
        suffixes = torch.empty((len(lengths), num_samples, k), device=self.device, dtype=self.observation_dtype)
        if suffixes.numel() == 0:
            return suffixes

        selected = torch.multinomial(component_probs, num_samples, replacement=True, generator=generator)
        for component_id, (component, (_, next_probs)) in enumerate(zip(self.components, summaries)):
            rows, draws = (selected == component_id).nonzero(as_tuple=True)
            if rows.numel():
                suffixes[rows, draws], _ = component._sample_paths(next_probs[rows], k, generator)
        return suffixes

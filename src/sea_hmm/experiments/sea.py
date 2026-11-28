from itertools import product
from math import log

import torch
from torch import Tensor

from .config import RunConfig
from .data import Episodes, Intent, episode_batches, observation_model, random_stream
from .models import SequenceClassifier
from .training import classifier_logits


def entropy(probabilities: Tensor) -> Tensor:
    return -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)


def _cross_entropy(probabilities: Tensor, log_predictions: Tensor) -> Tensor:
    return -torch.where(probabilities > 0, probabilities * log_predictions, 0).sum(dim=-1)


def _log_mean(log_probabilities: Tensor, dim: int) -> Tensor:
    return log_probabilities.logsumexp(dim=dim) - log(log_probabilities.shape[dim])


def oracle_terms(
    intent_posterior: Tensor,
    target_probs: Tensor,
    log_predictions: Tensor,
    extended_probs: Tensor,
    weights: Tensor,
) -> dict[str, Tensor]:
    """Evaluate the oracle decomposition and conditional clarification gain."""
    short_probs = intent_posterior @ target_probs
    short_entropy = entropy(short_probs)
    residual = intent_posterior @ entropy(target_probs)
    specification = short_entropy - residual
    total = _cross_entropy(short_probs, log_predictions)
    gains = _cross_entropy(extended_probs, short_probs[:, None].log()) - entropy(extended_probs)
    lower_bound = (weights * gains).sum(dim=-1)
    return {
        "residual": residual,
        "specification": specification,
        "excess_risk": total - short_entropy,
        "total": total,
        "specification_lower_bound": lower_bound,
        "specification_gap": specification - lower_bound,
    }


def continuation_terms(log_predictions: Tensor, weights: Tensor) -> dict[str, Tensor]:
    """Evaluate the canonical plug-in estimator for [episode, member, suffix, target] log probabilities."""
    log_member = (log_predictions + weights[:, None, :, None].log()).logsumexp(dim=2)
    log_ensemble = _log_mean(log_member, dim=1)
    residual = (entropy(log_predictions.exp()) * weights[:, None]).sum(dim=2).mean(dim=1)
    member_entropy = entropy(log_member.exp()).mean(dim=1)
    total = entropy(log_ensemble.exp())
    return {
        "effective_residual": residual,
        "specification_lower_bound": member_entropy - residual,
        "epistemic_disagreement": total - member_entropy,
        "total": total,
    }


def enumeration_size(intent: Intent, config: RunConfig) -> int:
    num_symbols = observation_model(intent).num_symbols
    if num_symbols is None:
        raise ValueError("exact enumeration requires discrete HMM observations")
    count = num_symbols ** config.data.lookahead
    if count > config.evaluation.max_enumerated_suffixes:
        raise ValueError(f"exact enumeration needs {count} suffixes, above evaluation.max_enumerated_suffixes")
    return count


def _log_predictions(
    members: list[SequenceClassifier], observations: Tensor, lengths: Tensor, anchors: Tensor | None,
    config: RunConfig,
) -> Tensor:
    return torch.stack([
        classifier_logits(model, observations, lengths, anchors, config.device).double().log_softmax(dim=-1)
        for model in members
    ], dim=1)


def _extensions(intent: Intent, batch: Episodes, config: RunConfig, rng, enumeration: Tensor | None):
    prefix_lengths = batch.prefix_lengths.to(config.device)
    observations = batch.observations.to(config.device)
    width = observations.shape[1]
    k = config.data.lookahead
    if enumeration is None:
        suffixes = intent.sample_continuations(
            observations, k, lengths=batch.prefix_lengths,
            num_samples=config.evaluation.num_continuations, generator=rng
        )
    else:
        suffixes = enumeration.expand(len(batch), -1, -1)
    count = suffixes.shape[1]
    prefix = observations.masked_fill(torch.arange(width, device=config.device) >= prefix_lengths[:, None], 0)
    extended = prefix[:, None].expand(-1, count, -1).clone()
    positions = prefix_lengths[:, None, None] + torch.arange(k, device=config.device)
    extended.scatter_(2, positions.expand_as(suffixes), suffixes)
    flat = extended.flatten(end_dim=1)
    lengths = (batch.prefix_lengths + k).repeat_interleave(count)
    anchors = None if batch.anchors is None else batch.anchors.repeat_interleave(count)
    if enumeration is None:
        weights = torch.full((len(batch), count), 1 / count, dtype=torch.float64, device=config.device)
    else:
        evidence = intent.log_likelihood(observations, lengths=batch.prefix_lengths)
        chunk_size = config.evaluation.batch_size
        full_evidence = torch.cat([
            intent.log_likelihood(flat[start:start + chunk_size], lengths=lengths[start:start + chunk_size])
            for start in range(0, flat.shape[0], chunk_size)
        ]).reshape_as(evidence[:, None].expand(-1, count))
        weights = torch.softmax(full_evidence - evidence[:, None], dim=-1)
        if not torch.isfinite(weights).all():
            raise ValueError("could not normalize enumerated continuation probabilities")
    return flat, lengths, anchors, weights


def _score_extensions(
    intent: Intent,
    members: list[SequenceClassifier],
    batch: Episodes,
    config: RunConfig,
    rng: torch.Generator | None,
    enumeration: Tensor | None,
) -> tuple[Tensor | None, Tensor | None, Tensor]:
    values, lengths, anchors, weights = _extensions(intent, batch, config, rng, enumeration)
    modes = config.evaluation.modes
    num_members = len(members) if "ensemble" in modes else int("single" in modes)
    num_targets = intent.target_probs.shape[1]
    oracle = weights.new_zeros((len(values), num_targets)) if "oracle" in modes else None
    estimates = weights.new_zeros((len(values), num_members, num_targets)) if num_members else None
    valid_indices = (weights.flatten() > 0).nonzero(as_tuple=True)[0]
    for indices in valid_indices.split(config.evaluation.batch_size):
        lengths_chunk = lengths[indices.cpu()]
        anchors_chunk = None if anchors is None else anchors[indices.cpu()]
        if oracle is not None:
            kwargs = {} if anchors_chunk is None else {"anchors": anchors_chunk}
            oracle[indices] = intent.predict_target(values[indices], lengths=lengths_chunk, **kwargs)
        if estimates is not None:
            estimates[indices] = _log_predictions(
                members[:num_members], values[indices], lengths_chunk, anchors_chunk, config
            )
    if oracle is not None:
        oracle = oracle.reshape(*weights.shape, num_targets)
    if estimates is not None:
        estimates = estimates.reshape(*weights.shape, num_members, num_targets).transpose(1, 2)
    return oracle, estimates, weights


def _classification_scores(log_probs: Tensor, targets: Tensor, view: str) -> dict[str, Tensor]:
    return {
        f"{view}_nll": -log_probs.gather(1, targets[:, None]).squeeze(1),
        f"{view}_accuracy": (log_probs.argmax(dim=-1) == targets).double(),
    }


@torch.no_grad()
def evaluate(intent: Intent, members: list[SequenceClassifier], test: Episodes, config: RunConfig):
    rng = random_stream(config.seed, "evaluation:continuations", config.device)
    method = config.evaluation.continuation_method
    enumeration = None
    if method == "enumerate":
        count = enumeration_size(intent, config)
        alphabet = range(observation_model(intent).num_symbols)
        enumeration = torch.tensor(list(product(alphabet, repeat=config.data.lookahead)), device=config.device)
    else:
        count = config.evaluation.num_continuations

    modes = config.evaluation.modes
    metrics = {"episodes": len(test), "continuations": {"method": method, "count": count}}
    columns: dict[str, list[Tensor]] = {}
    target_probs = intent.target_probs
    for batch in episode_batches(test, config.evaluation.batch_size):
        short = _log_mean(_log_predictions(members, batch.observations, batch.prefix_lengths, batch.anchors, config), 1)
        long = _log_mean(_log_predictions(members, batch.observations, batch.lengths, batch.anchors, config), 1)
        targets = batch.targets.to(config.device)
        groups = {"classifier": {
            **_classification_scores(short, targets, "prefix"),
            **_classification_scores(long, targets, "full"),
        }}
        oracle, estimates, weights = _score_extensions(intent, members, batch, config, rng, enumeration)
        if "oracle" in modes:
            kwargs = {} if batch.anchors is None else {"anchors": batch.anchors}
            posterior = intent.intent_posterior(batch.observations, lengths=batch.prefix_lengths, **kwargs)
            groups["oracle"] = oracle_terms(posterior, target_probs, short, oracle, weights)
        if "ensemble" in modes:
            groups["ensemble"] = continuation_terms(estimates, weights)
        if "single" in modes:
            groups["single"] = continuation_terms(estimates[:, :1], weights)

        for group, terms in groups.items():
            for term, values in terms.items():
                columns.setdefault(f"{group}.{term}", []).append(values.cpu())
    episodes = {name: torch.cat(values) for name, values in columns.items()}
    for name, values in episodes.items():
        group, term = name.split(".", 1)
        metrics.setdefault(group, {})[term] = float(values.mean())
    return metrics, episodes

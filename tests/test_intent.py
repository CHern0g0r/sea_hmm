from itertools import product
from pathlib import Path

import pytest
import torch

from sea_hmm.config import load_experiment, load_hmm
from sea_hmm.data import AnchoredIntent, StaticIntent
from sea_hmm.hmm import StateEmittingHMM


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
INITIAL = (0.5, 0.5)
TRANSITION = ((0.8, 0.2), (0.1, 0.9))
EMISSIONS = ((0.75, 0.25), (0.25, 0.75))
TARGETS = ((0.9, 0.1), (0.2, 0.8))
PRIOR = (0.3, 0.7)
SECOND_EMISSIONS = (0.4, 0.6)


def _experiment(kind, device="cpu"):
    first = StateEmittingHMM(TRANSITION, categorical=EMISSIONS, initial_probs=INITIAL, device=device)
    if kind == "anchored":
        return AnchoredIntent(first, TARGETS, anchor=0)
    second = StateEmittingHMM([[1]], categorical=[SECOND_EMISSIONS], device=device)
    return StaticIntent((first, second), PRIOR, TARGETS)


def _joint_at_first_state(observations):
    joint = torch.zeros(2, dtype=torch.float64)
    for path in product(range(2), repeat=len(observations)):
        weight = INITIAL[path[0]] * EMISSIONS[path[0]][observations[0]]
        for source, destination, observation in zip(path, path[1:], observations[1:]):
            weight *= TRANSITION[source][destination] * EMISSIONS[destination][observation]
        joint[path[0]] += weight
    return joint


def _joint_intent(kind, observations):
    first_state_joint = _joint_at_first_state(observations)
    if kind == "anchored":
        return first_state_joint
    second_likelihood = 1.0
    for observation in observations:
        second_likelihood *= SECOND_EMISSIONS[observation]
    return torch.stack((PRIOR[0] * first_state_joint.sum(), first_state_joint.new_tensor(
        PRIOR[1] * second_likelihood
    )))


@pytest.mark.parametrize("kind", ("anchored", "static"))
def test_exact_intent_posterior_and_conditional_suffix_distribution(kind):
    experiment = _experiment(kind)
    kwargs = {"anchors": 0} if kind == "anchored" else {}
    prefix = torch.tensor([0])
    suffixes = torch.tensor(list(product(range(2), repeat=2)))
    extensions = torch.cat((prefix.expand(len(suffixes), -1), suffixes), dim=-1)
    prefix_joint = _joint_intent(kind, prefix.tolist())
    extended_joint = torch.stack([_joint_intent(kind, row.tolist()) for row in extensions])
    targets = torch.tensor(TARGETS, dtype=torch.float64)
    prefix_posterior = prefix_joint / prefix_joint.sum()
    extended_posterior = extended_joint / extended_joint.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(experiment.intent_posterior(prefix, **kwargs)[0], prefix_posterior)
    torch.testing.assert_close(experiment.intent_posterior(extensions, **kwargs), extended_posterior)

    log_weights = experiment.log_likelihood(extensions) - experiment.log_likelihood(prefix)
    weights = log_weights.exp()
    torch.testing.assert_close(weights, extended_joint.sum(dim=-1) / prefix_joint.sum())
    torch.testing.assert_close(weights.sum(), weights.new_tensor(1.0))
    predictions = experiment.predict_target(extensions, **kwargs)
    expected_prediction = prefix_posterior @ targets
    torch.testing.assert_close(predictions, extended_posterior @ targets)
    torch.testing.assert_close(experiment.predict_target(prefix, **kwargs)[0], expected_prediction)
    torch.testing.assert_close(weights @ predictions, expected_prediction)

    samples = experiment.sample_continuations(prefix, 2, num_samples=4096, generator=_seeded())
    suffix_ids = samples[0, :, 0] * 2 + samples[0, :, 1]
    observed_weights = torch.bincount(suffix_ids, minlength=4).to(weights.dtype) / suffix_ids.numel()
    torch.testing.assert_close(observed_weights, weights, atol=0.035, rtol=0)


def _seeded(device="cpu"):
    return torch.Generator(device=device).manual_seed(827)


def test_anchored_pairs_carry_one_target_and_anchor():
    model = _experiment("anchored").hmm
    experiment = AnchoredIntent(model, [[0, 1], [1, 0]], anchor="prefix_end")
    pair = experiment.sample([1, 3, 2], 2, generator=_seeded())
    assert pair.prefix_lengths.tolist() == [1, 3, 2]
    assert pair.trajectory.lengths.tolist() == [3, 5, 4]
    assert pair.anchors.tolist() == [0, 2, 1]
    torch.testing.assert_close(pair.latent_ids, pair.trajectory.hidden_states[torch.arange(3), pair.anchors])
    torch.testing.assert_close(pair.targets, 1 - pair.latent_ids)
    with pytest.raises(ValueError, match="within its unclarified prefix"):
        experiment.sample([0], 2)
    with pytest.raises(ValueError, match="valid emitting-state position"):
        experiment.intent_posterior(torch.tensor([0, 1]), anchors=1, lengths=[1])


def test_static_component_persists_in_episodes_and_suffixes():
    components = [
        StateEmittingHMM([[1]], categorical=[row], device="cpu") for row in ((1, 0, 0), (0, 1, 0))
    ]
    experiment = StaticIntent(components, (0.5, 0.5), ((0, 1), (1, 0)))
    pair = experiment.sample([1, 3] * 32, 2, generator=_seeded())
    assert pair.anchors is None
    assert pair.latent_ids.unique().tolist() == [0, 1]
    valid = torch.arange(5) < pair.trajectory.lengths[:, None]
    torch.testing.assert_close(pair.trajectory.observations[valid], pair.latent_ids[:, None].expand_as(valid)[valid])
    torch.testing.assert_close(pair.targets, 1 - pair.latent_ids)

    empty = torch.empty(0, dtype=torch.long)
    torch.testing.assert_close(experiment.intent_posterior(empty)[0], torch.tensor([0.5, 0.5], dtype=torch.float64))
    suffixes = experiment.sample_continuations(empty, 4, num_samples=64, generator=_seeded())[0]
    assert suffixes[:, 0].unique().tolist() == [0, 1]
    assert (suffixes == suffixes[:, :1]).all()

    # The second component loses all support at the first observation.
    prefix = torch.tensor([0, 0, 0])
    assert experiment.intent_posterior(prefix).tolist() == [[1.0, 0.0]]
    assert (experiment.sample_continuations(prefix, 3, num_samples=4, generator=_seeded()) == 0).all()
    impossible = torch.tensor([2, 2])
    assert experiment.log_likelihood(impossible).isneginf().all()
    with pytest.raises(ValueError, match="zero likelihood under the mixture"):
        experiment.intent_posterior(impossible)
    with pytest.raises(ValueError, match="zero likelihood under the mixture"):
        experiment.sample_continuations(impossible, 2)


def test_hmm_continuations_transition_after_the_prefix():
    model = StateEmittingHMM(
        [[0, 1], [1, 0]], categorical=[[1, 0], [0, 1]], initial_probs=[1, 0], device="cpu"
    )
    prefixes = torch.tensor([[0, -1], [0, 1], [-1, -1]])
    lengths = [1, 2, 0]
    expected = torch.tensor([[[1, 0, 1]], [[0, 1, 0]], [[0, 1, 0]]])
    samples = model.sample_continuations(prefixes, 3, lengths=lengths, generator=_seeded())
    torch.testing.assert_close(samples, expected)
    assert model.sample_continuations(prefixes, 0, lengths=lengths).shape == (3, 1, 0)


@pytest.mark.parametrize("name", ("mess3", "cpg_island", "gaussian_regime_switching"))
def test_process_yaml_examples(name):
    model = load_hmm(EXAMPLES / f"{name}.yaml", device="cpu", dtype=torch.float32)
    batch = model.sample([4, 2], generator=_seeded())
    assert model.filter(batch.observations, lengths=batch.lengths).dtype == torch.float32
    assert model.log_likelihood(batch.observations, lengths=batch.lengths).isfinite().all()


@pytest.mark.parametrize("kind", ("anchored", "static"))
def test_experiment_yaml_examples(kind, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    experiment = load_experiment(EXAMPLES / f"{kind}_mess3.yaml", device="cpu")
    pair = experiment.sample([2, 4], 2, generator=_seeded())
    kwargs = {"anchors": pair.anchors} if kind == "anchored" else {}
    prediction = experiment.predict_target(pair.trajectory.observations, lengths=pair.prefix_lengths, **kwargs)
    torch.testing.assert_close(prediction.sum(dim=-1), prediction.new_ones(2))
    suffixes = experiment.sample_continuations(
        pair.trajectory.observations, 2, lengths=pair.prefix_lengths, num_samples=3, generator=_seeded()
    )
    assert suffixes.shape == (2, 3, 2)


@pytest.mark.parametrize("kind", ("anchored", "static"))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_intent_cuda(kind):
    gpu = _experiment(kind, "cuda:0")
    cpu = _experiment(kind)
    pair = gpu.sample([1, 3], 2, generator=_seeded("cuda:0"))
    again = gpu.sample([1, 3], 2, generator=_seeded("cuda:0"))
    values = pair.trajectory.observations
    for key in ("prefix_lengths", "latent_ids", "targets"):
        assert getattr(pair, key).is_cuda
        torch.testing.assert_close(getattr(pair, key), getattr(again, key))
    assert values.is_cuda and pair.trajectory.hidden_states.is_cuda
    torch.testing.assert_close(values, again.trajectory.observations)
    kwargs = {"anchors": pair.anchors} if kind == "anchored" else {}
    cpu_kwargs = {"anchors": pair.anchors.cpu()} if kind == "anchored" else {}
    actual = gpu.intent_posterior(values, lengths=pair.prefix_lengths, **kwargs)
    expected = cpu.intent_posterior(values.cpu(), lengths=pair.prefix_lengths.cpu(), **cpu_kwargs)
    torch.testing.assert_close(actual.cpu(), expected)
    suffixes = gpu.sample_continuations(
        values, 2, lengths=pair.prefix_lengths, num_samples=4096, generator=_seeded("cuda:0")
    )
    assert suffixes.is_cuda
    first_prefix = values[0, :1].cpu()
    extensions = torch.tensor([[first_prefix.item(), 0], [first_prefix.item(), 1]])
    expected_first = (cpu.log_likelihood(extensions) - cpu.log_likelihood(first_prefix)).exp()[1]
    torch.testing.assert_close(suffixes[0, :, 0].double().mean().cpu(), expected_first, atol=0.04, rtol=0)

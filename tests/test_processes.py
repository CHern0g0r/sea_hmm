from itertools import product
from math import exp, pi, sqrt
from pathlib import Path

import pytest
import torch
import yaml

from sea_hmm.config import load_hmm
from sea_hmm.hmm import StateEmittingHMM


EXAMPLES = Path(__file__).resolve().parents[1] / "cfg" / "process"
PROCESSES = ("mess3", "cpg_island", "gaussian_regime_switching")


def _write_config(path, config):
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def enumerate_posterior(observations, initial, transition, emission):
    num_states = len(initial)
    posterior = torch.zeros(len(observations), num_states, dtype=torch.float64)
    likelihood = 0.0
    for path in product(range(num_states), repeat=len(observations)):
        weight = initial[path[0]] * emission(path[0], observations[0])
        for source, destination, observation in zip(path, path[1:], observations[1:]):
            weight *= transition(source, destination) * emission(destination, observation)
        likelihood += weight
        for t, state in enumerate(path):
            posterior[t, state] += weight
    return posterior / likelihood, torch.tensor(likelihood, dtype=torch.float64)


def cpg_transition(source, destination):
    region, previous_base = divmod(source, 4)
    next_region, next_base = divmod(destination, 4)
    region_probability = ((0.999, 0.001), (0.01, 0.99))[region][next_region]
    base_tables = (
        ((300, 205, 285, 210), (322, 298, 78, 302), (248, 246, 298, 208), (177, 239, 292, 292)),
        ((180, 274, 426, 120), (171, 368, 274, 188), (161, 339, 375, 125), (79, 355, 384, 182)),
    )
    row = base_tables[next_region][previous_base]
    return region_probability * row[next_base] / sum(row)


def gaussian_emission(state, observation):
    mean, std = ((-1.0, 0.6), (1.0, 1.3))[state]
    return exp(-0.5 * ((observation - mean) / std) ** 2) / (std * sqrt(2 * pi))


@pytest.mark.parametrize(
    "process,kwargs,observations,initial,transition,emission",
    [
        (
            "mess3", {}, (0, 2, 1), (1 / 3,) * 3,
            lambda i, j: 0.9 if i == j else 0.05,
            lambda i, symbol: 0.85 if i == symbol else 0.075,
        ),
        (
            "cpg_island", {}, (1, 2, 1), (0, 0.3, 0, 0, 0, 0.7, 0, 0),
            cpg_transition, lambda i, symbol: float(i % 4 == symbol),
        ),
        (
            "gaussian_regime_switching",
            {"transition_probs": [[0.8, 0.2], [0.1, 0.9]], "emissions": [[-1, 0.6], [1, 1.3]]},
            (-0.4, 1.2, 0.7),
            (0.8, 0.2),
            lambda i, j: ((0.8, 0.2), (0.1, 0.9))[i][j],
            gaussian_emission,
        ),
    ],
)
def test_posteriors_match_hidden_path_enumeration(
    process, kwargs, observations, initial, transition, emission, tmp_path
):
    config = yaml.safe_load((EXAMPLES / f"{process}.yaml").read_text(encoding="utf-8"))
    config.update(kwargs, initial_probs=list(initial))
    model = load_hmm(_write_config(tmp_path / "hmm.yaml", config), device="cpu")
    values = torch.tensor(observations, dtype=torch.float64)
    smoothed = model.smooth(values)[0]
    filtered = model.filter(values)[0]
    expected, likelihood = enumerate_posterior(observations, initial, transition, emission)
    torch.testing.assert_close(smoothed, expected)
    torch.testing.assert_close(model.log_likelihood(values)[0].exp(), likelihood)
    for length, belief in enumerate(filtered, start=1):
        expected, _ = enumerate_posterior(observations[:length], initial, transition, emission)
        torch.testing.assert_close(belief, expected[-1])
    torch.testing.assert_close(filtered[-1], smoothed[-1])


@pytest.mark.parametrize("process", PROCESSES)
def test_ragged_batches_ignore_padding_and_keep_exact_paths(process):
    model = load_hmm(EXAMPLES / f"{process}.yaml", device="cpu", dtype=torch.float32)
    lengths = (2, 0, 1, 3, 2)
    first = model.sample(lengths, generator=torch.Generator().manual_seed(42))
    second = model.sample(lengths, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(first.observations, second.observations)
    torch.testing.assert_close(first.hidden_states, second.hidden_states)
    assert first.observations.shape == (5, 3)
    assert first.hidden_states.shape == (5, 3)
    assert first.lengths.tolist() == list(lengths)

    padded = first.observations.clone()
    for row, length in enumerate(lengths):
        assert (first.observations[row, length:] == 0).all()
        assert (first.hidden_states[row, length:] == -1).all()
        states = first.hidden_states[row, :length]
        assert ((states >= 0) & (states < model.num_states)).all()
        padded[row, length:] = -100 if padded.dtype == torch.long else float("nan")

    for track in (model.filter, model.smooth):
        beliefs = track(padded, lengths=first.lengths)
        assert beliefs.shape == (5, 3, model.num_states)
        assert beliefs.dtype == torch.float32
        for row, length in enumerate(lengths):
            actual = beliefs[row, :length]
            torch.testing.assert_close(actual, track(first.observations[row, :length])[0])
            torch.testing.assert_close(actual.sum(dim=-1), torch.ones(length))
            assert torch.count_nonzero(beliefs[row, length:]) == 0

    log_likelihood = model.log_likelihood(padded, lengths=first.lengths)
    assert log_likelihood[1] == 0
    for row, length in enumerate(lengths):
        torch.testing.assert_close(log_likelihood[row:row + 1], model.log_likelihood(padded[row, :length]))


def test_each_observation_matches_its_emitting_state():
    generator = torch.Generator().manual_seed(72)
    cpg = load_hmm(EXAMPLES / "cpg_island.yaml", device="cpu").sample(64, generator=generator)
    torch.testing.assert_close(cpg.observations, cpg.hidden_states % 4)

    gaussian = StateEmittingHMM(
        ((0, 1), (1, 0)), gaussian=((-1, 1), (0.01, 0.01)), initial_probs=(1, 0), device="cpu"
    ).sample(64, generator=generator)
    torch.testing.assert_close(gaussian.hidden_states[0], torch.arange(64) % 2)
    assert ((gaussian.observations > 0) == (gaussian.hidden_states == 1)).all()


def test_known_start_and_impossible_observations():
    identity = torch.eye(3)
    model = StateEmittingHMM(identity, categorical=identity, initial_probs=(0, 1, 0), device="cpu")
    observations = torch.tensor([1, 1])
    expected = torch.tensor([[[0.0, 1.0, 0.0]] * 2], dtype=torch.float64)
    for track in (model.filter, model.smooth):
        torch.testing.assert_close(track(observations), expected)
        torch.testing.assert_close(track(observations[:0]), expected[:, :0])
        with pytest.raises(ValueError, match="likelihood"):
            track(torch.tensor([1, 0, 1]))
        with pytest.raises(ValueError, match="categorical observations"):
            track(torch.tensor([1.5]))
        with pytest.raises(ValueError, match="lengths"):
            track(observations, lengths=[3])
    with pytest.raises(ValueError, match="lengths"):
        model.sample([1, -1])
    with pytest.raises(ValueError, match="unique stationary"):
        StateEmittingHMM(identity, categorical=identity, device="cpu")
    with pytest.raises(ValueError, match="Gaussian observations"):
        gaussian = load_hmm(EXAMPLES / "gaussian_regime_switching.yaml", device="cpu")
        gaussian.filter(torch.tensor([float("inf")]))
    assert model.log_likelihood(torch.tensor([1, 0, 1])).isneginf().all()


def test_config_controls_state_count_transitions_and_emissions(tmp_path):
    config = {
        "type": "discrete",
        "name": "Mess3",
        "num_states": 2,
        "transition_probs": [[0, 1], [1, 0]],
        "emissions": [[0, 0, 1], [1, 0, 0]],
        "initial_probs": [1, 0],
    }
    path = _write_config(tmp_path / "custom.yaml", config)
    model = load_hmm(path, device="cpu")
    batch = model.sample(4, generator=torch.Generator().manual_seed(42))
    assert model.name == "Mess3" and model.num_states == 2 and model.num_symbols == 3
    assert batch.hidden_states.tolist() == [[0, 1, 0, 1]]
    assert batch.observations.tolist() == [[2, 0, 2, 0]]

    config["name"] = "Any name"
    renamed = load_hmm(_write_config(path, config), device="cpu")
    assert renamed.name == "Any name"
    torch.testing.assert_close(renamed.smooth(batch.observations), model.smooth(batch.observations))


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"num_states": 3}, "transition_probs must have shape"),
        ({"emissions": [[1, 0]]}, "emissions must have 2 rows"),
        ({"emissions": [[1, 0], [0.9, 0]]}, "categorical probabilities.*sum to one"),
        ({"type": "continuous", "emissions": [[0, 1], [1, 0]]}, "positive finite variances"),
        ({"type": "continuous", "emissions": [[0], [1]]}, "mean, standard_deviation"),
    ],
)
def test_invalid_config_matrices(tmp_path, overrides, error):
    config = {
        "type": "discrete",
        "num_states": 2,
        "transition_probs": [[0, 1], [1, 0]],
        "emissions": [[1, 0], [0, 1]],
    }
    config.update(overrides)
    with pytest.raises(ValueError, match=error):
        load_hmm(_write_config(tmp_path / "hmm.yaml", config), device="cpu")


@pytest.mark.parametrize("process", PROCESSES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_sampling_filtering_and_smoothing(process):
    path = EXAMPLES / f"{process}.yaml"
    gpu = load_hmm(path, device="cuda:0")
    cpu = load_hmm(path, device="cpu")
    batch = gpu.sample([0, 1, 20, 3], generator=torch.Generator(device=gpu.device).manual_seed(42))
    assert batch.observations.is_cuda and batch.hidden_states.is_cuda and batch.lengths.is_cuda
    for gpu_track, cpu_track in ((gpu.filter, cpu.filter), (gpu.smooth, cpu.smooth)):
        beliefs = gpu_track(batch.observations, lengths=batch.lengths)
        assert beliefs.is_cuda
        torch.testing.assert_close(
            beliefs.cpu(), cpu_track(batch.observations.cpu(), lengths=batch.lengths.cpu())
        )

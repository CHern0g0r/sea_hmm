from itertools import product
from math import exp, pi, sqrt

import pytest
import torch

from sea_hmm.hmm import CpGIslandModel, GaussianRegimeSwitchingHMM, Mess3


PROCESSES = (Mess3, CpGIslandModel, GaussianRegimeSwitchingHMM)


def enumerate_posterior(observations, initial, edge_weight):
    num_states = len(initial)
    posterior = torch.zeros(len(observations) + 1, num_states, dtype=torch.float64)
    for path in product(range(num_states), repeat=len(observations) + 1):
        weight = initial[path[0]]
        for source, destination, observation in zip(path, path[1:], observations):
            weight *= edge_weight(source, destination, observation)
        for t, state in enumerate(path):
            posterior[t, state] += weight
    return posterior / posterior.sum(dim=-1, keepdim=True)


def mess3_edge(source, destination, observation):
    # Shai et al., Appendix A.3. The matrices for the other symbols are rotations.
    symbol_zero = (
        (0.765, 0.00375, 0.00375),
        (0.0425, 0.0675, 0.00375),
        (0.0425, 0.00375, 0.0675),
    )
    return symbol_zero[(source - observation) % 3][(destination - observation) % 3]


def cpg_edge(source, destination, observation):
    if destination % 4 != observation:
        return 0.0
    region, previous_base = divmod(source, 4)
    next_region = destination // 4
    region_probability = ((0.999, 0.001), (0.01, 0.99))[region][next_region]
    base_tables = (
        ((300, 205, 285, 210), (322, 298, 78, 302), (248, 246, 298, 208), (177, 239, 292, 292)),
        ((180, 274, 426, 120), (171, 368, 274, 188), (161, 339, 375, 125), (79, 355, 384, 182)),
    )
    row = base_tables[next_region][previous_base]
    return region_probability * row[observation] / sum(row)


def gaussian_edge(source, destination, observation):
    mean, std = ((-1.0, 0.6), (1.0, 1.3))[source]
    transition = ((0.8, 0.2), (0.1, 0.9))[source][destination]
    return transition * exp(-0.5 * ((observation - mean) / std) ** 2) / (std * sqrt(2 * pi))


@pytest.mark.parametrize(
    "process,kwargs,observations,initial,edge_weight",
    [
        (Mess3, {}, (0, 2, 1), (1 / 3,) * 3, mess3_edge),
        (CpGIslandModel, {}, (1, 2, 1), (0.3, 0, 0, 0, 0.7, 0, 0, 0), cpg_edge),
        (
            GaussianRegimeSwitchingHMM,
            {"transition_probs": ((0.8, 0.2), (0.1, 0.9)), "stds": (0.6, 1.3)},
            (-0.4, 1.2, 0.7),
            (0.8, 0.2),
            gaussian_edge,
        ),
    ],
)
def test_posteriors_match_hidden_path_enumeration(process, kwargs, observations, initial, edge_weight):
    model = process(initial_probs=initial, device="cpu", **kwargs)
    values = torch.tensor(observations, dtype=torch.float64)
    smoothed = model.smooth(values)[0]
    filtered = model.filter(values)[0]
    torch.testing.assert_close(smoothed, enumerate_posterior(observations, initial, edge_weight))
    for length, belief in enumerate(filtered):
        expected = enumerate_posterior(observations[:length], initial, edge_weight)[-1]
        torch.testing.assert_close(belief, expected)
    torch.testing.assert_close(filtered[-1], smoothed[-1])
    assert not torch.allclose(filtered[0], smoothed[0])


@pytest.mark.parametrize("process", PROCESSES)
def test_ragged_batches_ignore_padding_and_keep_exact_paths(process):
    model = process(device="cpu", dtype=torch.float32)
    lengths = (2, 0, 1, 3, 2)
    first = model.sample(lengths, generator=torch.Generator().manual_seed(42))
    second = model.sample(lengths, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(first.observations, second.observations)
    torch.testing.assert_close(first.hidden_states, second.hidden_states)
    assert first.observations.shape == (5, 3)
    assert first.hidden_states.shape == (5, 4)
    assert first.lengths.tolist() == list(lengths)

    padded = first.observations.clone()
    for row, length in enumerate(lengths):
        assert (first.observations[row, length:] == 0).all()
        assert (first.hidden_states[row, length + 1:] == -1).all()
        assert ((first.hidden_states[row, :length + 1] >= 0)
                & (first.hidden_states[row, :length + 1] < model.num_states)).all()
        padded[row, length:] = -100 if padded.dtype == torch.long else float("nan")

    for track in (model.filter, model.smooth):
        beliefs = track(padded, lengths=first.lengths)
        assert beliefs.shape == (5, 4, model.num_states)
        assert beliefs.dtype == torch.float32
        for row, length in enumerate(lengths):
            actual = beliefs[row, :length + 1]
            torch.testing.assert_close(actual, track(first.observations[row, :length])[0])
            torch.testing.assert_close(actual.sum(dim=-1), torch.ones(length + 1))
            assert torch.count_nonzero(beliefs[row, length + 1:]) == 0


def test_sampling_respects_source_and_destination_states():
    generator = torch.Generator().manual_seed(72)
    mess3 = Mess3(0.5, 1, initial_probs=(1, 0, 0), device="cpu").sample(64, generator=generator)
    assert mess3.hidden_states[0, 0] == 0
    assert (mess3.hidden_states[:, :-1] != mess3.hidden_states[:, 1:]).all()
    torch.testing.assert_close(mess3.observations, mess3.hidden_states[:, 1:])

    cpg = CpGIslandModel(1, 1, device="cpu").sample(64, generator=generator)
    assert (cpg.hidden_states[:, :-1] // 4 != cpg.hidden_states[:, 1:] // 4).all()
    torch.testing.assert_close(cpg.observations, cpg.hidden_states[:, 1:] % 4)

    gaussian = GaussianRegimeSwitchingHMM(
        ((0, 1), (1, 0)), means=(-1, 1), stds=(0.01, 0.01), initial_probs=(1, 0), device="cpu"
    ).sample(64, generator=generator)
    torch.testing.assert_close(gaussian.hidden_states[0], torch.arange(65) % 2)
    assert ((gaussian.observations > 0) == (gaussian.hidden_states[:, :-1] == 1)).all()


def test_known_start_and_impossible_observations():
    model = Mess3(0, 1, initial_probs=(0, 1, 0), device="cpu")
    observations = torch.tensor([1, 1])
    expected = torch.tensor([[[0.0, 1.0, 0.0]] * 3], dtype=torch.float64)
    for track in (model.filter, model.smooth):
        torch.testing.assert_close(track(observations), expected)
        torch.testing.assert_close(track(observations[:0]), expected[:, :1])
        with pytest.raises(ValueError, match="likelihood"):
            track(torch.tensor([1, 0, 1]))
        with pytest.raises(ValueError, match="categorical observations"):
            track(torch.tensor([1.5]))
        with pytest.raises(ValueError, match="lengths"):
            track(observations, lengths=[3])
    with pytest.raises(ValueError, match="lengths"):
        model.sample([1, -1])
    with pytest.raises(ValueError, match="unique stationary"):
        Mess3(0, device="cpu")
    with pytest.raises(ValueError, match="Gaussian observations"):
        GaussianRegimeSwitchingHMM(device="cpu").filter(torch.tensor([float("inf")]))


@pytest.mark.parametrize("process", PROCESSES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_sampling_filtering_and_smoothing(process):
    gpu = process(device="cuda:0")
    cpu = process(device="cpu")
    batch = gpu.sample([0, 1, 20, 3], generator=torch.Generator(device=gpu.device).manual_seed(42))
    assert batch.observations.is_cuda and batch.hidden_states.is_cuda and batch.lengths.is_cuda
    for gpu_track, cpu_track in ((gpu.filter, cpu.filter), (gpu.smooth, cpu.smooth)):
        beliefs = gpu_track(batch.observations, lengths=batch.lengths)
        assert beliefs.is_cuda
        torch.testing.assert_close(
            beliefs.cpu(), cpu_track(batch.observations.cpu(), lengths=batch.lengths.cpu())
        )

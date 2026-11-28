import csv
import json
import os
import subprocess
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import pytest
import torch
import yaml

from sea_hmm.config import _build_experiment
from sea_hmm.data import AnchoredIntent
from sea_hmm.experiments.config import ModelConfig, load_run_config
from sea_hmm.experiments.data import Episodes, generate_batches, load_dataset, materialize_split, save_dataset
from sea_hmm.experiments.models import SequenceClassifier
from sea_hmm.experiments.sea import _score_extensions, continuation_terms, entropy, oracle_terms
from sea_hmm.hmm import StateEmittingHMM


RECIPES = Path(__file__).resolve().parents[1] / "cfg" / "experiments"


def _small_recipe(tmp_path, name, mode, device="cpu"):
    original = RECIPES / f"{name}.yaml"
    raw = yaml.safe_load(original.read_text(encoding="utf-8"))
    raw.update(intent=str((original.parent / raw["intent"]).resolve()), output_dir="output", device=device)
    raw["data"].update(
        mode=mode, dataset_path="episodes.pt", prefix_lengths=[3, 5], train_size=12, validation_size=5, test_size=4
    )
    raw["training"].update(epochs=1, batch_size=4, ensemble_size=2)
    raw["evaluation"].update(batch_size=3, num_continuations=5)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _cli(tmp_path, command, config_path):
    result = subprocess.run(
        [sys.executable, "-m", "sea_hmm.experiments", command, str(config_path)],
        cwd=tmp_path, env={**os.environ, "OMP_NUM_THREADS": "1"}, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_run(tmp_path, config_path):
    _cli(tmp_path, "run", config_path)
    output = tmp_path / "output"
    resolved = load_run_config(output / "resolved.yaml")
    original = load_run_config(config_path)
    assert resolved == original
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["episodes"] == original.data.test_size
    method = original.evaluation.continuation_method
    assert metrics["continuations"] == {"method": method, "count": 27 if method == "enumerate" else 5}
    assert metrics["single"]["epistemic_disagreement"] == 0
    assert metrics["oracle"]["total"] == pytest.approx(
        sum(metrics["oracle"][term] for term in ("residual", "specification", "excess_risk"))
    )
    assert metrics["ensemble"]["total"] == pytest.approx(sum(metrics["ensemble"][term] for term in (
        "effective_residual", "specification_lower_bound", "epistemic_disagreement"
    )))
    history = json.loads((output / "training_history.json").read_text(encoding="utf-8"))
    assert len(history) == 2 and history[0]["seed"] != history[1]["seed"]
    for member in range(2):
        checkpoint = torch.load(output / "checkpoints" / f"member_{member}.pt", weights_only=True)
        assert checkpoint["validation_nll"] == history[member]["validation_nll"]
        assert checkpoint["model"]["output.1.bias"].numel() == len(original.intent["target_probs"][0])
    with (output / "per_episode.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == original.data.test_size
    assert all("oracle.specification_gap" in row for row in rows)
    if original.intent["type"] == "anchored":
        assert all(int(row["anchor"]) == int(row["prefix_length"]) - 1 for row in rows)
    else:
        assert all(row["anchor"] == "" for row in rows)


@pytest.mark.parametrize("name", ("static_mess3_transformer", "anchored_gaussian_lstm"))
@pytest.mark.parametrize("mode", ("pre_generated", "online"))
def test_recipe_cli(tmp_path, name, mode):
    path = _small_recipe(tmp_path, name, mode)
    if mode == "pre_generated":
        _cli(tmp_path, "generate", path)
    _assert_run(tmp_path, path)
    assert (tmp_path / "episodes.pt").exists() == (mode == "pre_generated")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("name", ("static_mess3_transformer", "anchored_gaussian_lstm"))
def test_cuda_recipe(tmp_path, name):
    _assert_run(tmp_path, _small_recipe(tmp_path, name, "online", "cuda"))


def test_dataset_round_trip_and_seeded_splits(tmp_path):
    config = load_run_config(_small_recipe(tmp_path, "anchored_gaussian_lstm", "pre_generated"))
    intent = _build_experiment(config.intent, device=config.device, dtype=torch.float64)
    save_dataset(intent, config)
    splits = load_dataset(config)
    for name, episodes in splits.items():
        recreated = materialize_split(intent, config, name)
        for field, stored in episodes.tensors().items():
            torch.testing.assert_close(stored, recreated.tensors()[field])
    assert not torch.equal(splits["train"].observations[:4], splits["test"].observations)

    def first_batch(epoch):
        return next(generate_batches(intent, config, 12, 4, f"training-epoch:{epoch}"))

    torch.testing.assert_close(first_batch(1).observations, first_batch(1).observations)
    assert not torch.equal(first_batch(1).observations, first_batch(2).observations)
    with pytest.raises(ValueError, match="incompatible HMM or data settings"):
        load_dataset(replace(config, seed=config.seed + 1))


@pytest.mark.parametrize("kind", ("transformer", "lstm"))
def test_classifier_padding_and_anchor(kind):
    torch.manual_seed(42)
    classifier = SequenceClassifier(
        ModelConfig(kind, width=8, layers=1, dropout=0), num_symbols=None, num_targets=3,
        max_length=5, anchored=True
    ).eval()
    prefix = torch.tensor([[0.5, -1.0, 2.0]])
    padded = torch.tensor([[0.5, -1.0, 2.0, float("nan"), float("nan")]])
    lengths, anchor = torch.tensor([3]), torch.tensor([1])
    with torch.no_grad():
        original = classifier(prefix, lengths, anchor)
        torch.testing.assert_close(classifier(padded, lengths, anchor), original)
        assert not torch.allclose(classifier(padded, lengths, torch.tensor([0])), original)


def test_oracle_decomposition_with_perfect_clarification():
    posterior = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    targets = torch.eye(2, dtype=torch.float64)
    prediction = torch.tensor([[0.6, 0.4]], dtype=torch.float64)
    terms = oracle_terms(posterior, targets, prediction.log(), targets[None], posterior)
    torch.testing.assert_close(terms["residual"], posterior.new_zeros(1))
    torch.testing.assert_close(terms["specification"], entropy(posterior))
    torch.testing.assert_close(terms["specification_lower_bound"], entropy(posterior))
    torch.testing.assert_close(terms["specification_gap"], posterior.new_zeros(1))
    torch.testing.assert_close(terms["total"], -prediction.log().mean(dim=-1))
    torch.testing.assert_close(terms["excess_risk"], terms["total"] - entropy(posterior))


def test_weighted_ensemble_decomposition_and_single_model_reduction():
    predictions = torch.tensor([[[[0.95, 0.05], [0.5, 0.5]], [[0.8, 0.2], [0.1, 0.9]]]], dtype=torch.float64)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    terms = continuation_terms(predictions.log(), weights)
    member_probs = (predictions * weights[:, None, :, None]).sum(dim=2)
    expected_residual = (entropy(predictions) * weights[:, None]).sum(dim=2).mean(dim=1)
    member_entropy = entropy(member_probs).mean(dim=1)
    ensemble_entropy = entropy(member_probs.mean(dim=1))
    torch.testing.assert_close(terms["effective_residual"], expected_residual)
    torch.testing.assert_close(terms["specification_lower_bound"], member_entropy - expected_residual)
    torch.testing.assert_close(terms["epistemic_disagreement"], ensemble_entropy - member_entropy)
    assert terms["epistemic_disagreement"].item() > 0
    single = continuation_terms(predictions[:, :1].log(), weights)
    assert single["epistemic_disagreement"].item() == 0
    torch.testing.assert_close(single["total"], single["effective_residual"] + single["specification_lower_bound"])


def test_weighted_enumeration_skips_impossible_suffixes(tmp_path):
    config = load_run_config(_small_recipe(tmp_path, "static_mess3_transformer", "online"))
    config = replace(
        config, data=replace(config.data, lookahead=2), evaluation=replace(config.evaluation, modes=["oracle"])
    )
    hmm = StateEmittingHMM([[1]], categorical=[[0.75, 0.25, 0]], device="cpu")
    intent = AnchoredIntent(hmm, [[1, 0]], anchor=0)
    batch = Episodes.from_sample(intent.sample([2, 3], 2), width=5)
    suffixes = list(product(range(3), repeat=2))
    oracle, _, weights = _score_extensions(intent, [], batch, config, None, torch.tensor(suffixes))
    probabilities = (0.75, 0.25, 0)
    expected = weights.new_tensor([probabilities[i] * probabilities[j] for i, j in suffixes])
    torch.testing.assert_close(weights, expected.expand(2, -1))
    assert (oracle[weights == 0] == 0).all()
    assert (oracle[..., 0][weights > 0] == 1).all()

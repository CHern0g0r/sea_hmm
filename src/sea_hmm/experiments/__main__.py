import argparse
import csv
import json
from pathlib import Path

import torch
import yaml

from ..config import _build_experiment
from .config import RunConfig, load_run_config
from .data import Intent, load_dataset, materialize_split, save_dataset
from .sea import enumeration_size, evaluate
from .training import train_members


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _run(intent: Intent, config: RunConfig) -> None:
    if config.evaluation.continuation_method == "enumerate":
        enumeration_size(intent, config)
    if config.data.mode == "pre_generated":
        splits = load_dataset(config)
        training, validation, test = (splits[name] for name in ("train", "validation", "test"))
    else:
        training = None
        validation = materialize_split(intent, config, "validation")
        test = materialize_split(intent, config, "test")

    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved.yaml").write_text(yaml.safe_dump(config.snapshot(), sort_keys=False), encoding="utf-8")
    members, history = train_members(intent, config, training, validation)
    _write_json(output / "training_history.json", history)
    metrics, scores = evaluate(intent, members, test, config)
    _write_json(output / "metrics.json", metrics)
    with (output / "per_episode.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("episode", "prefix_length", "target", "latent_id", "anchor", *scores))
        anchors = [""] * len(test) if test.anchors is None else test.anchors.tolist()
        writer.writerows(zip(
            range(len(test)), test.prefix_lengths.tolist(), test.targets.tolist(), test.latent_ids.tolist(),
            anchors, *(values.tolist() for values in scores.values())
        ))
    print(f"Metrics written to {output / 'metrics.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate datasets and run SEA HMM experiments.")
    parser.add_argument("command", choices=("generate", "run"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        config = load_run_config(args.config)
        intent = _build_experiment(config.intent, device=config.device, dtype=torch.float64)
        if args.command == "generate":
            print(f"Dataset written to {save_dataset(intent, config)}", flush=True)
        else:
            _run(intent, config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

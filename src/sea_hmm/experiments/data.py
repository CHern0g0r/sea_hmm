import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch import Tensor
from torch.nn.functional import pad

from ..data import AnchoredIntent, IntentBatch, StaticIntent
from .config import RunConfig


Intent = AnchoredIntent | StaticIntent
SPLITS = ("train", "validation", "test")
GENERATION_BATCH_SIZE = 256


def observation_model(intent: Intent):
    return intent.hmm if isinstance(intent, AnchoredIntent) else intent.components[0]


def stream_seed(seed: int, stream: str) -> int:
    digest = hashlib.sha256(f"{seed}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "little") >> 1


def random_stream(seed: int, stream: str, device="cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(stream_seed(seed, stream))


@dataclass(frozen=True)
class Episodes:
    observations: Tensor
    hidden_states: Tensor
    lengths: Tensor
    prefix_lengths: Tensor
    latent_ids: Tensor
    targets: Tensor
    anchors: Tensor | None

    @classmethod
    def from_sample(cls, batch: IntentBatch, width: int) -> "Episodes":
        trajectory = batch.trajectory
        padding = (0, width - trajectory.observations.shape[1])
        return cls(
            pad(trajectory.observations, padding),
            pad(trajectory.hidden_states, padding, value=-1),
            trajectory.lengths, batch.prefix_lengths, batch.latent_ids, batch.targets, batch.anchors
        )

    def __len__(self) -> int:
        return self.targets.numel()

    def tensors(self) -> dict[str, Tensor | None]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def select(self, indices) -> "Episodes":
        return Episodes(**{
            name: None if values is None else values[indices] for name, values in self.tensors().items()
        })

    def cpu(self) -> "Episodes":
        return Episodes(**{
            name: None if values is None else values.cpu() for name, values in self.tensors().items()
        })


def generate_batches(intent: Intent, config: RunConfig, size: int, batch_size: int, stream: str):
    prefix_rng = random_stream(config.seed, f"{stream}:length")
    hmm_rng = random_stream(config.seed, f"{stream}:hmm", observation_model(intent).device)
    choices = torch.tensor(config.data.prefix_lengths)
    width = max(config.data.prefix_lengths) + config.data.lookahead
    for start in range(0, size, batch_size):
        count = min(batch_size, size - start)
        lengths = choices[torch.randint(len(choices), (count,), generator=prefix_rng)]
        batch = intent.sample(lengths, config.data.lookahead, generator=hmm_rng)
        yield Episodes.from_sample(batch, width)


def materialize_split(intent: Intent, config: RunConfig, split: str) -> Episodes:
    batches = [
        batch.cpu().tensors() for batch in generate_batches(
            intent, config, getattr(config.data, f"{split}_size"), GENERATION_BATCH_SIZE, split
        )
    ]
    return Episodes(**{
        name: None if values is None else torch.cat([batch[name] for batch in batches])
        for name, values in batches[0].items()
    })


def episode_batches(episodes: Episodes, batch_size: int, order: Tensor | None = None):
    for start in range(0, len(episodes), batch_size):
        selection = slice(start, start + batch_size)
        yield episodes.select(selection if order is None else order[selection])


def save_dataset(intent: Intent, config: RunConfig) -> Path:
    path = config.data.dataset_path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {split: materialize_split(intent, config, split).tensors() for split in SPLITS}
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp_path = Path(stream.name)
        try:
            torch.save({
                "metadata": config.dataset_metadata(),
                "generation_device": str(observation_model(intent).device),
                "splits": data,
            }, stream)
            stream.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return path


def load_dataset(config: RunConfig) -> dict[str, Episodes]:
    path = config.data.dataset_path
    if not path.is_file():
        raise ValueError(f"dataset {path} is missing; run the generate command first")
    dataset = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(dataset, dict) or dataset.get("metadata") != config.dataset_metadata():
        raise ValueError(f"dataset {path} was generated with incompatible HMM or data settings")
    splits = dataset.get("splits", {})
    result = {}
    width = max(config.data.prefix_lengths) + config.data.lookahead
    for name in SPLITS:
        try:
            episodes = Episodes(**splits[name])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"dataset {path} has an invalid {name} split") from exc
        count = getattr(config.data, f"{name}_size")
        for field, values in episodes.tensors().items():
            if field == "anchors" and config.intent["type"] == "static" and values is None:
                continue
            shape = (count, width) if field in ("observations", "hidden_states") else (count,)
            if not isinstance(values, Tensor) or values.shape != shape:
                raise ValueError(f"dataset {path} has invalid {name}.{field} shape")
        if not ((episodes.lengths == episodes.prefix_lengths + config.data.lookahead).all()
                and torch.isin(episodes.prefix_lengths, torch.tensor(config.data.prefix_lengths)).all()):
            raise ValueError(f"dataset {path} has invalid {name} lengths")
        result[name] = episodes
    return result

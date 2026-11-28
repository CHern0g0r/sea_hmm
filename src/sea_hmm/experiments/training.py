import torch
from torch import Tensor
from torch.nn.functional import cross_entropy

from .config import RunConfig
from .data import (
    Episodes,
    Intent,
    episode_batches,
    generate_batches,
    observation_model,
    random_stream,
    stream_seed,
)
from .models import SequenceClassifier


def classifier_logits(
    model: SequenceClassifier, observations: Tensor, lengths: Tensor, anchors: Tensor | None, device: torch.device
) -> Tensor:
    width = int(lengths.max())
    return model(observations[:, :width].to(device), lengths, anchors)


def _pair_loss(model: SequenceClassifier, batch: Episodes, device: torch.device) -> Tensor:
    short = classifier_logits(model, batch.observations, batch.prefix_lengths, batch.anchors, device)
    long = classifier_logits(model, batch.observations, batch.lengths, batch.anchors, device)
    targets = batch.targets.to(device)
    return (cross_entropy(short, targets) + cross_entropy(long, targets)) / 2


def _training_batches(intent: Intent, config: RunConfig, episodes: Episodes | None, epoch: int):
    batch_size = config.training.batch_size
    stream = f"training-epoch:{epoch}"
    if episodes is None:
        yield from generate_batches(intent, config, config.data.train_size, batch_size, stream)
    else:
        order = torch.randperm(len(episodes), generator=random_stream(config.seed, stream))
        yield from episode_batches(episodes, batch_size, order)


def train_members(
    intent: Intent, config: RunConfig, training: Episodes | None, validation: Episodes
) -> tuple[list[SequenceClassifier], list[dict]]:
    source_model = observation_model(intent)
    model_args = {
        "num_symbols": source_model.num_symbols,
        "num_targets": intent.target_probs.shape[1],
        "max_length": max(config.data.prefix_lengths) + config.data.lookahead,
        "anchored": config.intent["type"] == "anchored",
    }
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    members, history = [], []
    for member_id in range(config.training.ensemble_size):
        seed = stream_seed(config.seed, f"member:{member_id}")
        torch.manual_seed(seed)
        model = SequenceClassifier(config.model, **model_args).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
        checkpoint = checkpoint_dir / f"member_{member_id}.pt"
        best_loss = float("inf")
        for epoch in range(1, config.training.epochs + 1):
            model.train()
            train_loss = torch.zeros((), device=config.device)
            for batch in _training_batches(intent, config, training, epoch):
                optimizer.zero_grad(set_to_none=True)
                loss = _pair_loss(model, batch, config.device)
                loss.backward()
                optimizer.step()
                train_loss += loss.detach() * len(batch)

            model.eval()
            total = torch.zeros((), device=config.device)
            with torch.no_grad():
                for batch in episode_batches(validation, config.training.batch_size):
                    total += _pair_loss(model, batch, config.device) * len(batch)
            train_nll = float(train_loss / config.data.train_size)
            validation_nll = float(total / len(validation))
            if not torch.isfinite(torch.tensor([train_nll, validation_nll])).all():
                raise ValueError(f"classifier member {member_id} has non-finite loss at epoch {epoch}")
            history.append({
                "member": member_id, "seed": seed, "epoch": epoch,
                "train_nll": train_nll, "validation_nll": validation_nll,
            })
            if validation_nll < best_loss:
                best_loss = validation_nll
                weights = {name: value.detach().cpu() for name, value in model.state_dict().items()}
                torch.save({"model": weights, "epoch": epoch, "validation_nll": best_loss, "seed": seed}, checkpoint)
            print(
                f"Member {member_id} epoch {epoch} train NLL {train_nll:.4f} val NLL {validation_nll:.4f}",
                flush=True,
            )
        best = torch.load(checkpoint, weights_only=True, map_location="cpu")
        model.load_state_dict(best["model"])
        members.append(model)
    return members, history

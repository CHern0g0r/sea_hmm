import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from .config import ModelConfig


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        num_symbols: int | None,
        num_targets: int,
        max_length: int,
        anchored: bool,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_symbols, config.width) if num_symbols is not None else None
        self.anchored = anchored
        input_width = (config.width if self.embedding is not None else 1) + int(anchored)
        self.transformer = config.type == "transformer"
        if self.transformer:
            self.input_projection = nn.Linear(input_width, config.width)
            self.position = nn.Embedding(max_length, config.width)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    config.width, config.heads, config.feedforward, config.dropout, batch_first=True
                ),
                config.layers,
                enable_nested_tensor=False,
            )
        else:
            self.encoder = nn.LSTM(
                input_width, config.width, config.layers, batch_first=True,
                dropout=config.dropout if config.layers > 1 else 0
            )
        self.output = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.width, num_targets))

    def forward(self, observations: Tensor, lengths: Tensor, anchors: Tensor | None = None) -> Tensor:
        positions = torch.arange(observations.shape[1], device=observations.device)
        padding = positions >= lengths.to(observations.device)[:, None]
        values = observations.masked_fill(padding, 0)
        if self.embedding is None:
            features = values.to(dtype=self.output[-1].weight.dtype).unsqueeze(-1)
        else:
            features = self.embedding(values)
        if self.anchored:
            if anchors is None:
                raise ValueError("anchored classifiers require anchor positions")
            marker = positions == anchors.to(observations.device)[:, None]
            features = torch.cat((features, marker.to(features.dtype).unsqueeze(-1)), dim=-1)

        if self.transformer:
            projected = self.input_projection(features) + self.position(positions)
            encoded = self.encoder(projected, src_key_padding_mask=padding)
            summary = encoded.masked_fill(padding.unsqueeze(-1), 0).sum(dim=1)
            summary = summary / lengths.to(observations.device)[:, None]
        else:
            packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.encoder(packed)
            summary = hidden[-1]
        return self.output(summary)

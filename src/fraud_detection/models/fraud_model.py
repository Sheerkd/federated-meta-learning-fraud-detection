"""Shared model architecture for all local, centralized, and federated methods."""

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    input_dim: int = 30
    hidden_1: int = 64
    hidden_2: int = 32
    dropout: float = 0.2

    def as_dict(self) -> dict:
        return asdict(self)


class FraudMLP(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.layers = nn.Sequential(
            nn.Linear(spec.input_dim, spec.hidden_1),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_1, spec.hidden_2),
            nn.ReLU(),
            nn.Linear(spec.hidden_2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)

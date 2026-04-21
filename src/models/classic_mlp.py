"""Classic MLP model for one-hot/tabular dense input."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any


class ClassicMLP(nn.Module):
    """
    MLP truyền thống cho dữ liệu đã One-Hot Encoding.

    Cấu trúc:
        Input
        -> Linear -> BatchNorm1d -> ReLU -> Dropout
        -> Linear -> BatchNorm1d -> ReLU -> Dropout
        -> Linear(1) -> Sigmoid
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        h1, h2 = hidden_dims

        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor shape [batch_size, input_dim]
        Returns:
            prob: xác suất lớp dương, shape [batch_size, 1]
        """
        return self.net(x)
    
def get_classic_mlp_search_space(input_dim: int) -> Dict[str, Any]:
    return {
        "input_dim": input_dim,
        "hidden_h1": [64, 128, 256],
        "hidden_h2": [32, 64, 128],
        "dropout": {"type": "float", "low": 0.1, "high": 0.5},
        "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
        "max_epochs": 100,
        "patience": 15,
    }

def build_classic_mlp(cfg: Dict[str, Any]) -> ClassicMLP:
    cfg = dict(cfg)  # copy
    input_dim = cfg.pop("input_dim", None)
    if input_dim is None:
        raise ValueError("input_dim required")

    h1 = int(cfg.get("hidden_h1", 128))
    h2 = int(cfg.get("hidden_h2", 64))
    hidden_dims = (h1, h2)

    return ClassicMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=float(cfg.get("dropout", 0.3)),
    )

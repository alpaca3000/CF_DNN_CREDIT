"""Classic MLP model for one-hot/tabular dense input."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


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

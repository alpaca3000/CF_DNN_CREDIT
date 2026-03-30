"""Embedding MLP model for mixed numeric + categorical tabular input."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _calc_emb_dim(cardinality: int) -> int:
    """
    Tính chiều embedding cho một biến categorical có `cardinality` nhãn.

    Quy tắc thường dùng (practical heuristic):
        emb_dim = min(50, (cardinality + 1) // 2)
    """
    return min(50, (cardinality + 1) // 2)


class EmbedMLP(nn.Module):
    """
    MLP có Entity Embeddings cho biến định danh + biến số.

    Input:
      - x_num: [batch_size, input_num_dim] (float)
      - x_cat: [batch_size, n_cat_features] (long/int), mỗi cột là id category của 1 biến

    Pipeline:
      1) Mỗi cột categorical đi qua 1 nn.Embedding riêng.
      2) Nối tất cả embedding vectors lại.
      3) Concatenate với x_num.
      4) Đưa qua 2 hidden layers + output sigmoid.
    """

    def __init__(
        self,
        input_num_dim: int,
        cat_dims: Sequence[int],
        emb_dims: Optional[Sequence[int]] = None,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_num_dim = input_num_dim
        self.cat_dims = list(cat_dims)
        self.n_cat = len(self.cat_dims)

        if emb_dims is None:
            self.emb_dims = [_calc_emb_dim(c) for c in self.cat_dims]
        else:
            if len(emb_dims) != self.n_cat:
                raise ValueError(
                    f"len(emb_dims)={len(emb_dims)} phải bằng len(cat_dims)={self.n_cat}"
                )
            self.emb_dims = list(emb_dims)

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=c, embedding_dim=e)
                for c, e in zip(self.cat_dims, self.emb_dims)
            ]
        )

        total_emb_dim = sum(self.emb_dims)
        mlp_input_dim = input_num_dim + total_emb_dim

        h1, h2 = hidden_dims
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, h1),
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

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_num: float tensor [batch_size, input_num_dim]
            x_cat: long tensor  [batch_size, n_cat_features]
        Returns:
            prob: xác suất lớp dương, shape [batch_size, 1]
        """
        if x_cat.size(1) != self.n_cat:
            raise ValueError(
                f"x_cat có {x_cat.size(1)} cột, nhưng model mong đợi {self.n_cat} cột."
            )

        emb_list = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        x_cat_emb = torch.cat(emb_list, dim=1) if emb_list else None

        if x_cat_emb is not None:
            x = torch.cat([x_num, x_cat_emb], dim=1)
        else:
            x = x_num

        return self.mlp(x)

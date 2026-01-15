from __future__ import annotations

import torch
import torch.nn as nn


class DRN(nn.Module):
    """
    Minimal DRN-style sequence model:
    GRU over time-binned incident activity → predicts next-step intensity.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, input_dim]
        returns: [B, 1] predicted next intensity
        """
        out, _h = self.rnn(x)
        last = out[:, -1, :]
        y = self.head(last)
        return y

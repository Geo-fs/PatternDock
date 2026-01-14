from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    Classic TextCNN for multi-label classification.

    Input: [batch, seq_len] token ids
    Output: [batch, num_labels] logits
    """

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        emb_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: tuple = (3, 4, 5),
        dropout: float = 0.2,
        pad_id: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)

        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=emb_dim, out_channels=num_filters, kernel_size=k) for k in kernel_sizes]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_labels)

        # init
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T]
        """
        emb = self.embedding(x)           # [B, T, E]
        emb = emb.transpose(1, 2)         # [B, E, T] for Conv1d

        pooled = []
        for conv in self.convs:
            h = F.relu(conv(emb))         # [B, F, T-k+1]
            p = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)  # [B, F]
            pooled.append(p)

        out = torch.cat(pooled, dim=1)    # [B, F * K]
        out = self.dropout(out)
        logits = self.fc(out)             # [B, num_labels]
        return logits

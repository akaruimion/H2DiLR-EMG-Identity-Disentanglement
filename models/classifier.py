"""
Gesture classification head for H2DiLR Stage 2.
"""

import torch
from torch import nn


class GestureClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                input_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.classifier(x)
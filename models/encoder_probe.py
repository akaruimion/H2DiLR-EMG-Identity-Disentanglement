"""
Gesture probe for the frozen Stage-1 H2D encoder.

Purpose
-------
Measure how much gesture information is already present
in the continuous encoder representation BEFORE H2D
vector quantization.

The encoder remains completely frozen.
Only the gesture classifier is trained.
"""

import torch
from torch import nn


class EncoderGestureProbe(nn.Module):
    """
    Frozen encoder + lightweight gesture classifier.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_tokens: int,
        latent_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.encoder = encoder

        # Freeze Stage-1 encoder.
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.encoder.eval()

        feature_dim = num_tokens * latent_dim

        # Keep this classifier intentionally simple.
        #
        # [B, num_tokens, latent_dim]
        #            ↓
        # flatten
        #            ↓
        # [B, num_tokens * latent_dim]
        #            ↓
        # 49 gesture logits
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.2),
            nn.Linear(
                feature_dim,
                num_classes,
            ),
        )

    def train(
        self,
        mode: bool = True,
    ):
        """
        Keep the encoder in evaluation mode even when
        the gesture classifier is training.
        """

        super().train(mode)

        self.encoder.eval()

        return self

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # We do not want gradients through Stage 1.
        with torch.no_grad():
            latent_tokens = self.encoder(x)

        logits = self.classifier(
            latent_tokens
        )

        return logits
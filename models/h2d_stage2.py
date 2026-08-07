"""
Stage-2 H2DiLR neural decoding model.
"""

from typing import Dict

import torch
from torch import nn

from models.h2dilr import H2DiLRStage1
from models.transformer import H2DTransformer
from models.classifier import GestureClassifier


class H2DiLRStage2(nn.Module):

    def __init__(
        self,
        stage1_model: H2DiLRStage1,
        latent_dim: int,
        transformer_dim: int,
        transformer_heads: int,
        transformer_layers: int,
        transformer_ffn_dim: int,
        transformer_dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.stage1 = stage1_model

        # Freeze encoder + quantizer.
        self.stage1.freeze_representation_modules()

        self.transformer = H2DTransformer(
            input_dim=latent_dim,
            model_dim=transformer_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            ffn_dim=transformer_ffn_dim,
            dropout=transformer_dropout,
        )

        self.classifier = GestureClassifier(
            input_dim=transformer_dim,
            num_classes=num_classes,
            dropout=transformer_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        # Frozen Stage-1 representation extraction.
        with torch.no_grad():

            latent_tokens = self.stage1.encoder(x)

            quantizer_output = self.stage1.quantizer(
                latent_tokens,
                subject_ids,
            )

            h2d_tokens = quantizer_output[
                "quantized"
            ]

        representation = self.transformer(
            h2d_tokens
        )

        logits = self.classifier(
            representation
        )

        return {
            "h2d_tokens": h2d_tokens,
            "representation": representation,
            "logits": logits,
        }
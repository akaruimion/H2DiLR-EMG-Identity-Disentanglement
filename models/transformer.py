"""
Transformer neural decoder for Stage-2 H2DiLR.

Input:
    [batch_size, num_tokens, latent_dim]

Output:
    [batch_size, transformer_dim]
"""

import torch
from torch import nn


class H2DTransformer(nn.Module):
    """
    Lightweight Transformer for downstream neural decoding.
    """

    def __init__(
        self,
        input_dim: int = 256,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.model_dim = model_dim

        self.input_projection = nn.Linear(
            input_dim,
            model_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.final_norm = nn.LayerNorm(
            model_dim
        )

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:

        if tokens.ndim != 3:
            raise ValueError(
                "Transformer input must have shape "
                "[batch, tokens, features]."
            )

        if tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected token dimension "
                f"{self.input_dim}, "
                f"received {tokens.shape[-1]}."
            )

        x = self.input_projection(
            tokens
        )

        x = self.transformer(
            x
        )

        x = self.final_norm(
            x
        )

        # Mean pooling over latent tokens
        x = x.mean(
            dim=1
        )

        return x
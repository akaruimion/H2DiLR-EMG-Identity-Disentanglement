"""
Reconstruction-only autoencoder control.

Uses exactly the H2D encoder and decoder,
but removes the H2D quantizer.
"""

import torch
from torch import nn
import torch.nn.functional as F

from models.encoder import H2DEncoder
from models.decoder import H2DDecoder


class ReconstructionAutoencoder(nn.Module):

    def __init__(
        self,
        input_channels=12,
        latent_dim=256,
        output_length=200,
    ):
        super().__init__()

        self.encoder = H2DEncoder(
            input_channels=input_channels,
            latent_dim=latent_dim,
        )

        self.decoder = H2DDecoder(
            latent_dim=latent_dim,
            output_channels=input_channels,
            output_length=output_length,
        )

    def forward(self, x):

        latent_tokens = self.encoder(x)

        reconstruction = self.decoder(
            latent_tokens
        )

        reconstruction_loss = F.mse_loss(
            reconstruction,
            x,
        )

        return {
            "latent_tokens": latent_tokens,
            "reconstruction": reconstruction,
            "reconstruction_loss": reconstruction_loss,
            "total_loss": reconstruction_loss,
        }
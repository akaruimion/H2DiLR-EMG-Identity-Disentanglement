"""
Stage-1 H2DiLR model.

This module combines:

1. H2DEncoder
2. H2DQuantizer
3. H2DDecoder

The model receives EMG windows and subject IDs, then returns
latent tokens, quantized tokens, reconstruction, and losses.
"""

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from models.encoder import H2DEncoder
from models.h2d_quantizer import H2DQuantizer
from models.decoder import H2DDecoder


class H2DiLRStage1(nn.Module):
    """
    Complete Stage-1 H2DiLR reconstruction model.
    """

    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        output_channels: int,
        output_length: int,
        num_subjects: int,
        shared_codebook_size: int,
        private_codebook_size: int,
        shared_ratio: float = 0.5,
        commitment_beta: float = 0.25,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()

        self.encoder = H2DEncoder(
            input_channels=input_channels,
            latent_dim=latent_dim,
        )

        self.quantizer = H2DQuantizer(
            num_subjects=num_subjects,
            shared_codebook_size=shared_codebook_size,
            private_codebook_size=private_codebook_size,
            embedding_dim=latent_dim,
            shared_ratio=shared_ratio,
            commitment_beta=commitment_beta,
            ema_decay=ema_decay,
            ema_epsilon=ema_epsilon,
        )

        self.decoder = H2DDecoder(
            latent_dim=latent_dim,
            output_channels=output_channels,
            output_length=output_length,
        )

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run a complete Stage-1 forward pass.

        Parameters
        ----------
        x:
            EMG windows shaped:

                [batch_size, time_steps, channels]

        subject_ids:
            Contiguous subject labels shaped:

                [batch_size]

        Returns
        -------
        dict
            Model outputs and Stage-1 losses.
        """

        latent_tokens = self.encoder(x)

        quantizer_output = self.quantizer(
            latent_tokens,
            subject_ids,
        )

        quantized_tokens = quantizer_output[
            "quantized"
        ]

        reconstruction = self.decoder(
            quantized_tokens
        )

        reconstruction_loss = F.mse_loss(
            reconstruction,
            x,
        )

        total_loss = (
            reconstruction_loss
            + quantizer_output["loss"]
        )

        return {
            "latent_tokens": latent_tokens,
            "quantized_tokens": quantized_tokens,
            "reconstruction": reconstruction,
            "reconstruction_loss": reconstruction_loss,
            "quantization_loss": quantizer_output["loss"],
            "private_codebook_loss": quantizer_output[
                "private_codebook_loss"
            ],
            "commitment_loss": quantizer_output[
                "commitment_loss"
            ],
            "total_loss": total_loss,
            "shared_mask": quantizer_output[
                "shared_mask"
            ],
            "private_mask": quantizer_output[
                "private_mask"
            ],
            "shared_indices": quantizer_output[
                "shared_indices"
            ],
            "private_indices": quantizer_output[
                "private_indices"
            ],
            "shared_perplexity": quantizer_output[
                "shared_perplexity"
            ],
            "private_perplexity": quantizer_output[
                "private_perplexity"
            ],
            "shared_tokens_per_sample": quantizer_output[
                "shared_tokens_per_sample"
            ],
            "private_tokens_per_sample": quantizer_output[
                "private_tokens_per_sample"
            ],
        }

    def freeze_representation_modules(self) -> None:
        """
        Freeze encoder and quantizer for Stage-2 neural decoding.
        """

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        for parameter in self.quantizer.parameters():
            parameter.requires_grad = False

        self.encoder.eval()
        self.quantizer.eval()

    def unfreeze_representation_modules(self) -> None:
        """
        Unfreeze encoder and quantizer.
        """

        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

        for parameter in self.quantizer.parameters():
            parameter.requires_grad = True
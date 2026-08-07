"""
Convolutional reconstruction decoder for H2DiLR.

Input shape:
    [batch_size, num_tokens, latent_dim]

Output shape:
    [batch_size, window_size, output_channels]
"""

import torch
from torch import nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """
    One-dimensional transpose-convolution decoder block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class H2DDecoder(nn.Module):
    """
    Decoder used during Stage-1 H2D pretraining.

    It reconstructs the original EMG window from quantized latent
    tokens.

    Parameters
    ----------
    latent_dim:
        Dimension of each quantized token.

    output_channels:
        Number of EMG channels to reconstruct.

    output_length:
        Required temporal length of the reconstructed signal.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        output_channels: int = 12,
        output_length: int = 200,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError(
                "latent_dim must be greater than zero."
            )

        if output_channels <= 0:
            raise ValueError(
                "output_channels must be greater than zero."
            )

        if output_length <= 0:
            raise ValueError(
                "output_length must be greater than zero."
            )

        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.output_length = output_length

        # Latent representation:
        # 256 channels, approximately 10 temporal tokens.

        self.block1 = DecoderBlock(
            in_channels=latent_dim,
            out_channels=512,
        )

        self.block2 = DecoderBlock(
            in_channels=512,
            out_channels=256,
        )

        self.block3 = DecoderBlock(
            in_channels=256,
            out_channels=128,
        )

        self.block4 = DecoderBlock(
            in_channels=128,
            out_channels=64,
        )

        # Convert decoder features back to the original EMG
        # channel count.
        self.output_projection = nn.Conv1d(
            in_channels=64,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct an EMG signal.

        Parameters
        ----------
        z:
            Quantized latent tokens with shape:

                [batch_size, num_tokens, latent_dim]

        Returns
        -------
        torch.Tensor
            Reconstructed EMG windows with shape:

                [batch_size, output_length, output_channels]
        """

        if z.ndim != 3:
            raise ValueError(
                "Decoder input must have shape "
                "[batch_size, num_tokens, latent_dim]. "
                f"Received shape: {tuple(z.shape)}"
            )

        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.latent_dim}, "
                f"but received {z.shape[-1]}."
            )

        # Input:
        # [batch, tokens, latent_dim]
        #
        # ConvTranspose1d expects:
        # [batch, channels, time]
        x = z.transpose(1, 2).contiguous()

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Guarantee that the temporal dimension exactly matches
        # the original window length.
        if x.shape[-1] != self.output_length:
            x = F.interpolate(
                x,
                size=self.output_length,
                mode="linear",
                align_corners=False,
            )

        x = self.output_projection(x)

        # Return:
        # [batch, time, channels]
        x = x.transpose(1, 2).contiguous()

        return x
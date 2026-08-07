"""
One-dimensional convolutional encoder for H2DiLR.

The encoder converts an EMG window with shape:

    [batch_size, time_steps, channels]

into a sequence of latent tokens with shape:

    [batch_size, num_tokens, latent_dim]
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    One-dimensional convolution block.

    The block applies:

        Conv1d
        GELU activation
        Optional average pooling
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_pooling: bool = True,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=1,
            padding=1,
        )

        # The paper specifies the convolution and pooling structure,
        # but does not explicitly identify the activation function.
        # GELU is used here as an implementation choice.
        self.activation = nn.GELU()

        if use_pooling:
            self.pool = nn.AvgPool1d(
                kernel_size=2,
                stride=2,
            )
        else:
            self.pool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)

        return x


class H2DEncoder(nn.Module):
    """
    CNN encoder used during the H2D pretraining stage.

    Parameters
    ----------
    input_channels:
        Number of signal channels in each input window.

    latent_dim:
        Dimension of each output latent token.
        The paper's default architecture uses 256.
    """

    def __init__(
        self,
        input_channels: int = 12,
        latent_dim: int = 256,
    ) -> None:
        super().__init__()

        if input_channels <= 0:
            raise ValueError(
                "input_channels must be greater than zero."
            )

        if latent_dim <= 0:
            raise ValueError(
                "latent_dim must be greater than zero."
            )

        # Paper architecture:
        # 4 x 1 convolution, 64 channels, stride 2.
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
        )

        # Block 1: 64 -> 128, then average pooling.
        self.block1 = ConvBlock(
            in_channels=64,
            out_channels=128,
            use_pooling=True,
        )

        # Block 2: 128 -> 256, then average pooling.
        self.block2 = ConvBlock(
            in_channels=128,
            out_channels=256,
            use_pooling=True,
        )

        # Block 3: 256 -> 512, then average pooling.
        self.block3 = ConvBlock(
            in_channels=256,
            out_channels=512,
            use_pooling=True,
        )

        # Block 4: 512 -> latent dimension, no pooling.
        self.block4 = ConvBlock(
            in_channels=512,
            out_channels=latent_dim,
            use_pooling=False,
        )

        self.input_channels = input_channels
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of EMG windows.

        Parameters
        ----------
        x:
            Tensor with shape:

                [batch_size, time_steps, channels]

        Returns
        -------
        torch.Tensor
            Latent token sequence with shape:

                [batch_size, num_tokens, latent_dim]
        """

        if x.ndim != 3:
            raise ValueError(
                "Encoder input must have three dimensions: "
                "[batch_size, time_steps, channels]. "
                f"Received shape: {tuple(x.shape)}"
            )

        if x.shape[-1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, "
                f"but received {x.shape[-1]}."
            )

        # Input:
        # [batch, time, channels]
        #
        # Conv1d expects:
        # [batch, channels, time]
        x = x.transpose(1, 2).contiguous()

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Current shape:
        # [batch, latent_dim, num_tokens]
        #
        # Vector quantization will expect:
        # [batch, num_tokens, latent_dim]
        x = x.transpose(1, 2).contiguous()

        return x
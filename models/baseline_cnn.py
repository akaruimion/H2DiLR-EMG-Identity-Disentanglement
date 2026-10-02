"""
Simple supervised CNN baseline for NinaPro DB2 gesture recognition.

Purpose
-------
This model intentionally does NOT use:
- H2D
- vector quantization
- shared/private codebooks
- reconstruction
- Transformer

It is used to determine whether the low gesture accuracy
comes from the basic EMG/data pipeline or from H2D.
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):

    def __init__(self, num_channels=12, num_classes=49):
        super().__init__()

        self.features = nn.Sequential(

            # Input:
            # [B, 12, 200]

            nn.Conv1d(
                in_channels=num_channels,
                out_channels=64,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # [B, 64, 100]

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # [B, 128, 50]

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # Convert variable temporal dimension
            # to one feature value per channel.
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):

        # Current DataLoader returns:
        # [B, 200, 12]
        #
        # Conv1d expects:
        # [B, channels, time]
        #
        # Therefore:
        # [B, 200, 12] -> [B, 12, 200]

        x = x.transpose(1, 2)

        x = self.features(x)

        logits = self.classifier(x)

        return logits


if __name__ == "__main__":

    model = BaselineCNN(
        num_channels=12,
        num_classes=49,
    )

    dummy_input = torch.randn(64, 200, 12)

    output = model(dummy_input)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)

    assert output.shape == (64, 49)

    print("Baseline CNN test passed.")
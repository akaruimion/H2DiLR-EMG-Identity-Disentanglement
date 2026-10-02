"""
Supervised diagnostic for H2DEncoder.

Purpose:
Determine whether the encoder architecture itself is capable
of learning gesture-discriminative EMG representations.

Unlike the encoder probe, the encoder is TRAINABLE here.
"""

import random
import numpy as np
import torch
import torch.nn as nn

from configs.config import Config
from datasets.dataloader import get_dataloaders
from models.encoder import H2DEncoder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SupervisedEncoderClassifier(nn.Module):

    def __init__(
        self,
        input_channels,
        latent_dim,
        num_classes,
        window_size,
    ):
        super().__init__()

        self.encoder = H2DEncoder(
            input_channels=input_channels,
            latent_dim=latent_dim,
        )

        # Determine output shape automatically.
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                window_size,
                input_channels,
            )

            latent = self.encoder(dummy)

        self.num_tokens = latent.shape[1]
        self.latent_dim = latent.shape[2]

        feature_dim = (
            self.num_tokens
            * self.latent_dim
        )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),

            nn.Dropout(0.2),

            nn.Linear(
                feature_dim,
                num_classes,
            ),
        )

    def forward(self, x):

        latent = self.encoder(x)

        logits = self.classifier(latent)

        return logits


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for X, y, _ in loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(
            logits,
            y,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * X.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == y
        ).sum().item()

        total += X.size(0)

    return (
        total_loss / total,
        100.0 * correct / total,
    )


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for X, y, _ in loader:

        X = X.to(device)
        y = y.to(device)

        logits = model(X)

        loss = criterion(
            logits,
            y,
        )

        total_loss += (
            loss.item() * X.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == y
        ).sum().item()

        total += X.size(0)

    return (
        total_loss / total,
        100.0 * correct / total,
    )


def main():

    print("=" * 65)
    print("Supervised H2DEncoder Diagnostic")
    print("=" * 65)

    set_seed(Config.SEED)

    device = Config.DEVICE

    print("Device:", device)

    print("\nLoading dataset...")

    train_loader, val_loader = (
        get_dataloaders()
    )

    model = SupervisedEncoderClassifier(
        input_channels=Config.NUM_CHANNELS,
        latent_dim=Config.LATENT_DIM,
        num_classes=Config.NUM_CLASSES,
        window_size=Config.WINDOW_SIZE,
    ).to(device)

    print()
    print(
        "Encoder latent tokens:",
        model.num_tokens,
    )

    print(
        "Latent dimension:",
        model.latent_dim,
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    criterion = nn.CrossEntropyLoss()

    # Much larger than Stage-1's 1e-5 because
    # this is a supervised diagnostic from scratch.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    epochs = 15

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )
    )

    best_val_accuracy = 0.0

    print()
    print("=" * 65)
    print("Training encoder directly on gesture labels")
    print("=" * 65)

    for epoch in range(
        1,
        epochs + 1,
    ):

        train_loss, train_acc = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        val_loss, val_acc = (
            validate(
                model,
                val_loader,
                criterion,
                device,
            )
        )

        scheduler.step()

        print()
        print(
            f"Epoch {epoch}/{epochs}"
        )

        print(
            f"Train loss:     "
            f"{train_loss:.6f}"
        )

        print(
            f"Train accuracy: "
            f"{train_acc:.2f}%"
        )

        print(
            f"Val loss:       "
            f"{val_loss:.6f}"
        )

        print(
            f"Val accuracy:   "
            f"{val_acc:.2f}%"
        )

        if val_acc > best_val_accuracy:

            best_val_accuracy = val_acc

            torch.save(
                model.state_dict(),
                "checkpoints/"
                "supervised_encoder_best.pt",
            )

            print(
                "Saved new best checkpoint"
            )

    print()
    print("=" * 65)
    print("Diagnostic Complete")
    print("=" * 65)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.2f}%"
    )

    print()
    print("Comparison:")
    print("Raw CNN baseline       : 76.12%")
    print("Frozen encoder probe   : 5.98%")

    print(
        "Supervised H2DEncoder : "
        f"{best_val_accuracy:.2f}%"
    )

    print("Full H2D              : ~5-8%")


if __name__ == "__main__":
    main()
"""
Train a simple supervised CNN baseline on NinaPro DB2.

Diagnostic purpose:
Determine whether the low H2D gesture accuracy is caused by
the data pipeline or by H2D representation learning.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn

from configs.config import Config
from datasets.dataloader import get_dataloaders
from models.baseline_cnn import BaselineCNN
from training.train_baseline import (
    train_baseline_one_epoch,
    validate_baseline,
)


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    print("=" * 60)
    print("NinaPro DB2 - Supervised CNN Baseline")
    print("=" * 60)

    set_seed(Config.SEED)

    device = Config.DEVICE

    print()
    print("Device:", device)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    print()
    print("Loading dataset...")

    train_loader, val_loader = get_dataloaders()

    print()
    print("Training batches  :", len(train_loader))
    print("Validation batches:", len(val_loader))

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = BaselineCNN(
        num_channels=Config.NUM_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    print()
    print(model)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Trainable parameters: "
        f"{total_parameters:,}"
    )

    # --------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # Diagnostic run.
    #
    # 15 epochs should already tell us whether
    # the supervised pipeline can learn.
    epochs = 15

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    # --------------------------------------------------
    # Checkpoint
    # --------------------------------------------------

    os.makedirs("checkpoints", exist_ok=True)

    checkpoint_path = (
        "checkpoints/baseline_cnn_best.pt"
    )

    best_val_accuracy = 0.0

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("Starting baseline training")
    print("=" * 60)

    for epoch in range(1, epochs + 1):

        train_loss, train_accuracy = (
            train_baseline_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
        )

        val_loss, val_accuracy = (
            validate_baseline(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
        )

        scheduler.step()

        print()
        print(f"Epoch {epoch}/{epochs}")

        print(
            f"Train loss:     "
            f"{train_loss:.6f}"
        )

        print(
            f"Train accuracy: "
            f"{train_accuracy:.2f}%"
        )

        print(
            f"Val loss:       "
            f"{val_loss:.6f}"
        )

        print(
            f"Val accuracy:   "
            f"{val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print(
                "Saved new best baseline checkpoint"
            )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("Baseline experiment complete")
    print("=" * 60)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.2f}%"
    )

    print()
    print(
        "Random-chance accuracy "
        "for 49 classes: ~2.04%"
    )

    print()
    print("Interpretation:")
    print(
        "If baseline accuracy is much higher than "
        "H2D (~5-8%), investigate H2D/Stage-1."
    )
    print(
        "If baseline accuracy is also very low, "
        "investigate preprocessing, labels, "
        "normalization, and data splitting."
    )


if __name__ == "__main__":
    main()
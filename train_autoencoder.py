"""
Train reconstruction-only encoder-decoder control.
"""

import random
from pathlib import Path

import numpy as np
import torch

from configs.config import Config
from datasets.dataloader import get_dataloaders
from models.reconstruction_autoencoder import (
    ReconstructionAutoencoder,
)
from training.train_autoencoder import (
    train_one_epoch,
    validate,
)


AUTOENCODER_EPOCHS = 10

# Keep identical to the H2D experiment
# for the first controlled comparison.
AUTOENCODER_LR = 5e-5


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    set_seed(Config.SEED)

    print("=" * 65)
    print("Reconstruction-Only Autoencoder Control")
    print("=" * 65)

    print("Device:", Config.DEVICE)
    print("Epochs:", AUTOENCODER_EPOCHS)
    print("Learning rate:", AUTOENCODER_LR)

    train_loader, val_loader = (
        get_dataloaders()
    )

    model = ReconstructionAutoencoder(
        input_channels=Config.NUM_CHANNELS,
        latent_dim=Config.LATENT_DIM,
        output_length=Config.WINDOW_SIZE,
    ).to(Config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=AUTOENCODER_LR,
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=AUTOENCODER_EPOCHS,
        )
    )

    checkpoint_path = (
        Path(Config.CHECKPOINT_DIR)
        / "reconstruction_autoencoder_best.pt"
    )

    best_val_loss = float("inf")

    for epoch in range(
        1,
        AUTOENCODER_EPOCHS + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            Config.DEVICE,
        )

        (
            val_loss,
            zero_loss,
            improvement,
        ) = validate(
            model,
            val_loader,
            Config.DEVICE,
        )

        scheduler.step()

        print()
        print(
            f"Epoch {epoch}/{AUTOENCODER_EPOCHS}"
        )

        print(
            f"Train reconstruction MSE: "
            f"{train_loss:.6f}"
        )

        print(
            f"Val reconstruction MSE:   "
            f"{val_loss:.6f}"
        )

        print(
            f"Zero-predictor MSE:        "
            f"{zero_loss:.6f}"
        )

        print(
            f"Improvement over zero:     "
            f"{improvement:.2f}%"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "best_validation_loss":
                        best_val_loss,
                },
                checkpoint_path,
            )

            print(
                "Saved new best checkpoint"
            )

    print()
    print("=" * 65)
    print("Autoencoder Control Complete")
    print("=" * 65)

    print(
        f"Best reconstruction MSE: "
        f"{best_val_loss:.6f}"
    )

    print()
    print(
        "H2D Stage-1 reconstruction MSE "
        "was approximately 1.077446"
    )

    print(
        "H2D improvement over zero "
        "was approximately 3.18%"
    )


if __name__ == "__main__":
    main()
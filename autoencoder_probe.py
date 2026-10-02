"""
Frozen gesture probe for the reconstruction-only autoencoder.

Purpose
-------
Test whether the encoder trained with reconstruction-only
pretraining at a stronger learning rate learned useful
gesture-discriminative features.

Only the classifier is trained.
The autoencoder encoder remains frozen.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from configs.config import Config
from datasets.dataloader import get_dataloaders
from models.reconstruction_autoencoder import (
    ReconstructionAutoencoder,
)
from models.encoder_probe import (
    EncoderGestureProbe,
)
from training.train_encoder_probe import (
    train_probe_one_epoch,
    validate_probe,
)


def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_autoencoder_checkpoint(
    model,
    checkpoint_path,
    device,
):

    print(
        "Loading autoencoder checkpoint:",
        checkpoint_path,
    )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    print(
        "Autoencoder checkpoint loaded successfully."
    )

    if isinstance(checkpoint, dict):

        if "epoch" in checkpoint:
            print(
                "Checkpoint epoch:",
                checkpoint["epoch"],
            )

        if "best_validation_loss" in checkpoint:
            print(
                "Best validation reconstruction loss:",
                checkpoint[
                    "best_validation_loss"
                ],
            )


def main():

    print("=" * 70)
    print(
        "Reconstruction Autoencoder Encoder Gesture Probe"
    )
    print("=" * 70)

    set_seed(
        Config.SEED
    )

    device = Config.DEVICE

    print(
        "Device:",
        device,
    )

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    train_loader, val_loader = (
        get_dataloaders()
    )

    print(
        "Training batches:",
        len(train_loader),
    )

    print(
        "Validation batches:",
        len(val_loader),
    )

    # --------------------------------------------------
    # Build autoencoder
    # --------------------------------------------------

    autoencoder = ReconstructionAutoencoder(
        input_channels=Config.NUM_CHANNELS,
        latent_dim=Config.LATENT_DIM,
        output_length=Config.WINDOW_SIZE,
    ).to(device)

    checkpoint_path = (
        Path(Config.CHECKPOINT_DIR)
        / "reconstruction_autoencoder_best.pt"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    load_autoencoder_checkpoint(
        model=autoencoder,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # --------------------------------------------------
    # Determine encoder output shape
    # --------------------------------------------------

    autoencoder.encoder.eval()

    dummy_input = torch.zeros(
        1,
        Config.WINDOW_SIZE,
        Config.NUM_CHANNELS,
        device=device,
    )

    with torch.no_grad():

        dummy_latent = (
            autoencoder.encoder(
                dummy_input
            )
        )

    num_tokens = (
        dummy_latent.shape[1]
    )

    latent_dim = (
        dummy_latent.shape[2]
    )

    print()
    print(
        "Encoder output shape:",
        tuple(dummy_latent.shape),
    )

    print(
        "Number of latent tokens:",
        num_tokens,
    )

    print(
        "Latent dimension:",
        latent_dim,
    )

    # --------------------------------------------------
    # Create frozen encoder probe
    # --------------------------------------------------

    model = EncoderGestureProbe(
        encoder=autoencoder.encoder,
        num_tokens=num_tokens,
        latent_dim=latent_dim,
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    encoder_trainable = sum(
        p.numel()
        for p in model.encoder.parameters()
        if p.requires_grad
    )

    classifier_trainable = sum(
        p.numel()
        for p in model.classifier.parameters()
        if p.requires_grad
    )

    print()
    print(
        "Trainable encoder parameters:",
        encoder_trainable,
    )

    print(
        "Trainable classifier parameters:",
        classifier_trainable,
    )

    assert encoder_trainable == 0

    print(
        "Encoder freeze check: PASSED"
    )

    # --------------------------------------------------
    # Training setup
    # --------------------------------------------------

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = (
        torch.optim.AdamW(
            model.classifier.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
    )

    epochs = 15

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )
    )

    best_val_accuracy = 0.0

    output_checkpoint = (
        Path(Config.CHECKPOINT_DIR)
        / "autoencoder_probe_best.pt"
    )

    # --------------------------------------------------
    # Train probe
    # --------------------------------------------------

    print()
    print("=" * 70)
    print(
        "Training gesture classifier on "
        "FROZEN autoencoder encoder"
    )
    print("=" * 70)

    for epoch in range(
        1,
        epochs + 1,
    ):

        (
            train_loss,
            train_accuracy,
        ) = train_probe_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        (
            val_loss,
            val_accuracy,
        ) = validate_probe(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
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

        if (
            val_accuracy
            > best_val_accuracy
        ):

            best_val_accuracy = (
                val_accuracy
            )

            torch.save(
                model.classifier.state_dict(),
                output_checkpoint,
            )

            print(
                "Saved new best "
                "autoencoder-probe checkpoint"
            )

    # --------------------------------------------------
    # Final comparison
    # --------------------------------------------------

    print()
    print("=" * 70)
    print(
        "Autoencoder Encoder Probe Complete"
    )
    print("=" * 70)

    print(
        "Best validation accuracy:",
        f"{best_val_accuracy:.2f}%",
    )

    print()
    print(
        "Current representation comparison:"
    )

    print(
        "Raw CNN baseline              : "
        "76.12%"
    )

    print(
        "H2D pretrained encoder probe : "
        "5.98%"
    )

    print(
        "Reconstruction encoder probe : "
        f"{best_val_accuracy:.2f}%"
    )

    print(
        "Supervised H2DEncoder         : "
        "92.27%"
    )

    print(
        "Full H2D                     : "
        "~5-8%"
    )

    print()

    if best_val_accuracy >= 30.0:

        print(
            "Interpretation:"
        )

        print(
            "Stronger reconstruction pretraining "
            "produced substantially more useful "
            "gesture features."
        )

        print(
            "Next: retrain full H2D Stage 1 "
            "with the stronger learning rate."
        )

    elif best_val_accuracy >= 10.0:

        print(
            "Interpretation:"
        )

        print(
            "Reconstruction pretraining improved "
            "the representation, but gesture "
            "information is still limited."
        )

        print(
            "Next: test stronger Stage-1 training "
            "and inspect the pretraining objective."
        )

    else:

        print(
            "Interpretation:"
        )

        print(
            "Better reconstruction did not produce "
            "a strongly gesture-discriminative "
            "representation."
        )

        print(
            "Reconstruction quality alone is "
            "therefore not enough."
        )


if __name__ == "__main__":
    main()
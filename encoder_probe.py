"""
Evaluate gesture information in the frozen H2D encoder.

Experiment
----------
Raw EMG
   ↓
Frozen pretrained H2D encoder
   ↓
Continuous latent tokens
   ↓
Simple gesture classifier
   ↓
49 gestures

This determines whether gesture information is already
lost BEFORE H2D vector quantization.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from configs.config import Config
from datasets.dataloader import get_dataloaders
from models.encoder_probe import EncoderGestureProbe
from pretrain import create_model
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


def load_stage1_checkpoint(
    model,
    checkpoint_path,
    device,
):
    """
    Load the pretrained Stage-1 checkpoint.

    This supports several common checkpoint formats.
    """

    print(
        "Loading Stage-1 checkpoint:",
        checkpoint_path,
    )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    # Case 1:
    # checkpoint itself is a raw state_dict.
    if (
        isinstance(checkpoint, dict)
        and checkpoint
        and all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        )
    ):

        state_dict = checkpoint

    # Case 2:
    # checkpoint created by training utility.
    elif (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
        and isinstance(
            checkpoint["model"],
            dict,
        )
    ):

        state_dict = checkpoint[
            "model"
        ]

    else:

        if isinstance(checkpoint, dict):
            print(
                "Checkpoint keys:",
                list(checkpoint.keys()),
            )

        raise RuntimeError(
            "Could not identify the model state "
            "inside the checkpoint."
        )

    model.load_state_dict(
        state_dict
    )

    print(
        "Stage-1 checkpoint loaded successfully."
    )

    if isinstance(checkpoint, dict):

        if "epoch" in checkpoint:
            print(
                "Checkpoint epoch:",
                checkpoint["epoch"],
            )

        if "best_validation_loss" in checkpoint:
            print(
                "Checkpoint best validation loss:",
                checkpoint[
                    "best_validation_loss"
                ],
            )


def main():

    print("=" * 65)
    print("H2DiLR Encoder-Only Gesture Probe")
    print("=" * 65)

    set_seed(Config.SEED)

    device = Config.DEVICE

    print()
    print("Device:", device)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    print()
    print("Loading dataset...")

    train_loader, val_loader = (
        get_dataloaders()
    )

    print()
    print(
        "Training batches  :",
        len(train_loader),
    )

    print(
        "Validation batches:",
        len(val_loader),
    )

    # --------------------------------------------------
    # Load Stage-1 model
    # --------------------------------------------------

    stage1 = create_model().to(device)

    checkpoint_path = Path(
        Config.CHECKPOINT_DIR
    ) / "h2d_stage1_latest.pt"

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Stage-1 checkpoint not found: "
            f"{checkpoint_path}"
        )

    load_stage1_checkpoint(
        model=stage1,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # --------------------------------------------------
    # Determine encoder output shape automatically
    # --------------------------------------------------

    stage1.encoder.eval()

    dummy_input = torch.zeros(
        1,
        Config.WINDOW_SIZE,
        Config.NUM_CHANNELS,
        device=device,
    )

    with torch.no_grad():

        dummy_latent = stage1.encoder(
            dummy_input
        )

    num_tokens = dummy_latent.shape[1]
    latent_dim = dummy_latent.shape[2]

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
    # Create probe
    # --------------------------------------------------

    model = EncoderGestureProbe(
        encoder=stage1.encoder,
        num_tokens=num_tokens,
        latent_dim=latent_dim,
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    # Verify that encoder is frozen.
    encoder_trainable = sum(
        parameter.numel()
        for parameter
        in model.encoder.parameters()
        if parameter.requires_grad
    )

    classifier_trainable = sum(
        parameter.numel()
        for parameter
        in model.classifier.parameters()
        if parameter.requires_grad
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

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
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

    best_validation_accuracy = 0.0

    checkpoint_output = Path(
        Config.CHECKPOINT_DIR
    ) / "encoder_probe_best.pt"

    # --------------------------------------------------
    # Train only the probe classifier
    # --------------------------------------------------

    print()
    print("=" * 65)
    print(
        "Training gesture probe on "
        "FROZEN encoder features"
    )
    print("=" * 65)

    for epoch in range(
        1,
        epochs + 1,
    ):

        train_loss, train_accuracy = (
            train_probe_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
        )

        val_loss, val_accuracy = (
            validate_probe(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
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
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                val_accuracy
            )

            torch.save(
                model.classifier.state_dict(),
                checkpoint_output,
            )

            print(
                "Saved new best "
                "encoder-probe checkpoint"
            )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("=" * 65)
    print("Encoder Probe Complete")
    print("=" * 65)

    print(
        "Best validation accuracy:",
        f"{best_validation_accuracy:.2f}%",
    )

    print()
    print("Current diagnostic comparison:")
    print()
    print(
        "Raw CNN baseline : 76.12%"
    )

    print(
        "Encoder probe    :",
        f"{best_validation_accuracy:.2f}%",
    )

    print(
        "Full H2D         : ~5-8%"
    )

    print()

    if best_validation_accuracy >= 40.0:

        print(
            "Interpretation:"
        )
        print(
            "The continuous encoder representation "
            "contains substantial gesture information."
        )
        print(
            "The next suspect is H2D quantization "
            "and/or the Stage-2 decoder."
        )

    elif best_validation_accuracy >= 15.0:

        print(
            "Interpretation:"
        )
        print(
            "The encoder contains some gesture "
            "information, but substantially less "
            "than the raw CNN baseline."
        )
        print(
            "Stage-1 representation learning "
            "likely needs improvement."
        )

    else:

        print(
            "Interpretation:"
        )
        print(
            "Gesture information is already very "
            "weak before H2D quantization."
        )
        print(
            "Investigate and retrain Stage-1 "
            "before changing the Transformer."
        )


if __name__ == "__main__":
    main()
"""
Train Stage-2 Transformer classifier using frozen H2D representations.
"""

import random

import numpy as np
import torch

from configs.config import Config
from datasets.dataloader import get_dataloaders

from models.h2dilr import H2DiLRStage1
from models.h2d_stage2 import H2DiLRStage2

from training.train_classifier import (
    train_classifier_one_epoch,
    validate_classifier,
)


def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_stage1_model():

    return H2DiLRStage1(
        input_channels=Config.NUM_CHANNELS,
        latent_dim=Config.LATENT_DIM,
        output_channels=Config.NUM_CHANNELS,
        output_length=Config.WINDOW_SIZE,
        num_subjects=Config.NUM_SUBJECTS,
        shared_codebook_size=(
            Config.SHARED_CODEBOOK_SIZE
        ),
        private_codebook_size=(
            Config.PRIVATE_CODEBOOK_SIZE
        ),
        shared_ratio=Config.SHARED_RATIO,
        commitment_beta=Config.COMMITMENT_BETA,
        ema_decay=Config.EMA_DECAY,
        ema_epsilon=Config.EMA_EPSILON,
    )


def main():

    set_seed(Config.SEED)

    print("Device:", Config.DEVICE)
    print(
        "Stage-2 epochs:",
        Config.FINETUNE_EPOCHS,
    )

    train_loader, val_loader = get_dataloaders()

    # ----------------------------------------------------------
    # Load pretrained Stage-1
    # ----------------------------------------------------------

    stage1 = create_stage1_model()

    checkpoint = torch.load(
        Config.STAGE1_CHECKPOINT,
        map_location=Config.DEVICE,
    )

    stage1.load_state_dict(
        checkpoint["model_state_dict"]
    )

    stage1 = stage1.to(Config.DEVICE)

    # ----------------------------------------------------------
    # Build Stage-2 model
    # ----------------------------------------------------------

    model = H2DiLRStage2(
        stage1_model=stage1,
        latent_dim=Config.LATENT_DIM,
        transformer_dim=Config.TRANSFORMER_DIM,
        transformer_heads=Config.TRANSFORMER_HEADS,
        transformer_layers=Config.TRANSFORMER_LAYERS,
        transformer_ffn_dim=(
            Config.TRANSFORMER_FFN_DIM
        ),
        transformer_dropout=(
            Config.TRANSFORMER_DROPOUT
        ),
        num_classes=Config.NUM_CLASSES,
    ).to(Config.DEVICE)

    # Only train unfrozen parameters.
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=Config.FINETUNE_LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.FINETUNE_EPOCHS,
    )

    best_validation_accuracy = 0.0

    for epoch in range(
        1,
        Config.FINETUNE_EPOCHS + 1,
    ):

        print(
            f"\nEpoch {epoch}/"
            f"{Config.FINETUNE_EPOCHS}"
        )

        train_metrics = train_classifier_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=Config.DEVICE,
        )

        validation_metrics = validate_classifier(
            model=model,
            data_loader=val_loader,
            device=Config.DEVICE,
        )

        scheduler.step()

        print(
            "Train loss: "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            "Train accuracy: "
            f"{100 * train_metrics['accuracy']:.2f}%"
        )

        print(
            "Validation loss: "
            f"{validation_metrics['loss']:.6f}"
        )

        print(
            "Validation accuracy: "
            f"{100 * validation_metrics['accuracy']:.2f}%"
        )

        print(
            "Learning rate: "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )

        if (
            validation_metrics["accuracy"]
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_metrics["accuracy"]
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "best_validation_accuracy": (
                        best_validation_accuracy
                    ),
                },
                Config.STAGE2_CHECKPOINT,
            )

            print(
                "Saved new best Stage-2 checkpoint:",
                Config.STAGE2_CHECKPOINT,
            )

    print("\nStage-2 training completed.")

    print(
        "Best validation accuracy:",
        f"{100 * best_validation_accuracy:.2f}%",
    )


if __name__ == "__main__":
    main()
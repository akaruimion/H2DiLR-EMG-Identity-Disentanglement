"""
Run Stage-1 self-supervised H2DiLR pretraining.
"""

import random
from pathlib import Path

import numpy as np
import torch

from configs.config import Config
from datasets.dataloader import get_dataloaders
from models.h2dilr import H2DiLRStage1
from training.train_h2d import (
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model() -> H2DiLRStage1:
    """
    Create the complete Stage-1 model.
    """

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


def main() -> None:
    set_seed(Config.SEED)

    print("Device:", Config.DEVICE)
    print(
        "Stage-1 epochs:",
        Config.PRETRAIN_EPOCHS,
    )

    train_loader, val_loader = get_dataloaders()

    model = create_model().to(Config.DEVICE)

    private_codebook_parameters = list(
        model.quantizer.private_codebooks.parameters()
    )

    private_parameter_ids = {
        id(parameter)
        for parameter in private_codebook_parameters
    }

    main_parameters = [
        parameter
        for parameter in model.parameters()
        if (
            parameter.requires_grad
            and id(parameter) not in private_parameter_ids
        )
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": main_parameters,
                "lr": Config.PRETRAIN_LEARNING_RATE,
            },
            {
                "params": private_codebook_parameters,
                "lr": Config.PRIVATE_CODEBOOK_LEARNING_RATE,
            },
        ],
        weight_decay=Config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.PRETRAIN_EPOCHS,
    )

    history = {
        "train_total_loss": [],
        "train_reconstruction_loss": [],
        "train_quantization_loss": [],
        "val_total_loss": [],
        "val_reconstruction_loss": [],
        "val_quantization_loss": [],
    }

    best_validation_loss = float("inf")

    best_checkpoint_path = Path(
        Config.CHECKPOINT_DIR
    ) / "h2d_stage1_best.pt"

    latest_checkpoint_path = Path(
        Config.CHECKPOINT_DIR
    ) / "h2d_stage1_latest.pt"

    for epoch in range(
        1,
        Config.PRETRAIN_EPOCHS + 1,
    ):
        print(
            f"\nEpoch {epoch}/"
            f"{Config.PRETRAIN_EPOCHS}"
        )

        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=Config.DEVICE,
            gradient_clip_norm=(
                Config.GRADIENT_CLIP_NORM
            ),
        )

        validation_metrics = validate_one_epoch(
            model=model,
            data_loader=val_loader,
            device=Config.DEVICE,
        )

        scheduler.step()

        history["train_total_loss"].append(
            train_metrics["total_loss"]
        )

        history[
            "train_reconstruction_loss"
        ].append(
            train_metrics["reconstruction_loss"]
        )

        history[
            "train_quantization_loss"
        ].append(
            train_metrics["quantization_loss"]
        )

        history["val_total_loss"].append(
            validation_metrics["total_loss"]
        )

        history[
            "val_reconstruction_loss"
        ].append(
            validation_metrics[
                "reconstruction_loss"
            ]
        )

        history[
            "val_quantization_loss"
        ].append(
            validation_metrics[
                "quantization_loss"
            ]
        )

        print(
            "Train total loss: "
            f"{train_metrics['total_loss']:.6f}"
        )

        print(
            "Train reconstruction loss: "
            f"{train_metrics['reconstruction_loss']:.6f}"
        )

        print(
            "Train quantization loss: "
            f"{train_metrics['quantization_loss']:.6f}"
        )

        print(
            "Train private-codebook loss: "
            f"{train_metrics['private_codebook_loss']:.6f}"
        )

        print(
            "Train commitment loss: "
            f"{train_metrics['commitment_loss']:.6f}"
        )

        print(
            "Validation total loss: "
            f"{validation_metrics['total_loss']:.6f}"
        )

        print(
            "Validation reconstruction loss: "
            f"{validation_metrics['reconstruction_loss']:.6f}"
        )

        print(
            "Validation quantization loss: "
            f"{validation_metrics['quantization_loss']:.6f}"
        )

        print(
            "Validation private-codebook loss: "
            f"{validation_metrics['private_codebook_loss']:.6f}"
        )

        print(
            "Validation commitment loss: "
            f"{validation_metrics['commitment_loss']:.6f}"
        )

        print(
            "Main learning rate: "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )

        print(
            "Private-codebook learning rate: "
            f"{optimizer.param_groups[1]['lr']:.8f}"
        )

        save_checkpoint(
            checkpoint_path=str(
                latest_checkpoint_path
            ),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_validation_loss=(
                best_validation_loss
            ),
            history=history,
        )

        if (
            validation_metrics["total_loss"]
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_metrics["total_loss"]
            )

            save_checkpoint(
                checkpoint_path=str(
                    best_checkpoint_path
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_validation_loss=(
                    best_validation_loss
                ),
                history=history,
            )

            print(
                "Saved new best checkpoint:",
                best_checkpoint_path,
            )

    print("\nStage-1 pretraining completed.")

    print(
        "Best validation loss:",
        best_validation_loss,
    )

    print(
        "Best checkpoint:",
        best_checkpoint_path,
    )


if __name__ == "__main__":
    main()
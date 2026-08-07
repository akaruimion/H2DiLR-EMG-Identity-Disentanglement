"""
Evaluate the trained H2DiLR Stage-2 model.
"""

import torch

from configs.config import Config
from datasets.dataloader import get_dataloaders

from models.h2dilr import H2DiLRStage1
from models.h2d_stage2 import H2DiLRStage2

from evaluation.evaluate import evaluate_model
from sklearn.metrics import classification_report
from evaluation.confusion_matrix import (
    save_confusion_matrix,
)


def create_stage1_model():

    return H2DiLRStage1(
        input_channels=(
            Config.NUM_CHANNELS
        ),
        latent_dim=(
            Config.LATENT_DIM
        ),
        output_channels=(
            Config.NUM_CHANNELS
        ),
        output_length=(
            Config.WINDOW_SIZE
        ),
        num_subjects=(
            Config.NUM_SUBJECTS
        ),
        shared_codebook_size=(
            Config.SHARED_CODEBOOK_SIZE
        ),
        private_codebook_size=(
            Config.PRIVATE_CODEBOOK_SIZE
        ),
        shared_ratio=(
            Config.SHARED_RATIO
        ),
        commitment_beta=(
            Config.COMMITMENT_BETA
        ),
        ema_decay=(
            Config.EMA_DECAY
        ),
        ema_epsilon=(
            Config.EMA_EPSILON
        ),
    )


def main():

    print(
        "Device:",
        Config.DEVICE,
    )

    _, val_loader = (
        get_dataloaders()
    )

    # ----------------------------------------------------------
    # Build Stage 1
    # ----------------------------------------------------------

    stage1 = (
        create_stage1_model()
    )

    stage1_checkpoint = (
        torch.load(
            Config.STAGE1_CHECKPOINT,
            map_location=(
                Config.DEVICE
            ),
        )
    )

    stage1.load_state_dict(
        stage1_checkpoint[
            "model_state_dict"
        ]
    )

    stage1 = stage1.to(
        Config.DEVICE
    )

    # ----------------------------------------------------------
    # Build Stage 2
    # ----------------------------------------------------------

    model = H2DiLRStage2(
        stage1_model=stage1,
        latent_dim=(
            Config.LATENT_DIM
        ),
        transformer_dim=(
            Config.TRANSFORMER_DIM
        ),
        transformer_heads=(
            Config.TRANSFORMER_HEADS
        ),
        transformer_layers=(
            Config.TRANSFORMER_LAYERS
        ),
        transformer_ffn_dim=(
            Config.TRANSFORMER_FFN_DIM
        ),
        transformer_dropout=(
            Config.TRANSFORMER_DROPOUT
        ),
        num_classes=(
            Config.NUM_CLASSES
        ),
    ).to(
        Config.DEVICE
    )

    # ----------------------------------------------------------
    # Load trained Stage 2 checkpoint
    # ----------------------------------------------------------

    stage2_checkpoint = (
        torch.load(
            Config.STAGE2_CHECKPOINT,
            map_location=(
                Config.DEVICE
            ),
        )
    )

    model.load_state_dict(
        stage2_checkpoint[
            "model_state_dict"
        ]
    )

    # ----------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------

    (
        metrics,
        targets,
        predictions,
    ) = evaluate_model(
        model=model,
        data_loader=(
            val_loader
        ),
        device=(
            Config.DEVICE
        ),
    )

    print("\nEvaluation Results")

    print(
        "Loss:",
        f"{metrics['loss']:.6f}",
    )

    print(
        "Accuracy:",
        f"{100 * metrics['accuracy']:.2f}%",
    )

    print(
        "Precision:",
        f"{100 * metrics['precision']:.2f}%",
    )

    print(
        "Recall:",
        f"{100 * metrics['recall']:.2f}%",
    )

    print(
        "F1 Score:",
        f"{100 * metrics['f1']:.2f}%",
    )

    print("\nPer-Class Classification Report")

    report = classification_report(
        targets,
        predictions,
        labels=list(range(Config.NUM_CLASSES)),
        zero_division=0,
    )

    print(report)
    # ----------------------------------------------------------
    # Save confusion matrix
    # ----------------------------------------------------------

    confusion_matrix_path = (
        "results/"
        "confusion_matrix.png"
    )

    save_confusion_matrix(
        targets=targets,
        predictions=predictions,
        output_path=(
            confusion_matrix_path
        ),
    )

    print(
        "\nConfusion matrix saved to:",
        confusion_matrix_path,
    )


if __name__ == "__main__":
    main()
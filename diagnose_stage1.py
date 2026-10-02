"""
Diagnostics for the pretrained H2DiLR Stage-1 model.

Checks:
1. Reconstruction MSE
2. Zero-predictor reconstruction baseline
3. Quantization error
4. Shared codebook utilization
5. Private codebook utilization
6. Shared/private token routing

This script does NOT train anything.
"""

from pathlib import Path

import torch

from configs.config import Config
from datasets.dataloader import get_dataloaders
from pretrain import create_model


# Use a subset for a quick CPU diagnostic.
# Set to None later to evaluate the entire validation set.
MAX_BATCHES = None


def calculate_perplexity(counts):
    counts = counts.float()

    total = counts.sum()

    if total == 0:
        return 0.0

    probabilities = counts / total

    probabilities = probabilities[
        probabilities > 0
    ]

    perplexity = torch.exp(
        -torch.sum(
            probabilities
            * torch.log(probabilities)
        )
    )

    return perplexity.item()


def load_checkpoint(
    model,
    checkpoint_path,
    device,
):
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

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    print(
        "Checkpoint loaded:",
        checkpoint_path,
    )

    if isinstance(checkpoint, dict):

        if "epoch" in checkpoint:
            print(
                "Checkpoint epoch:",
                checkpoint["epoch"],
            )

        if "best_validation_loss" in checkpoint:
            print(
                "Best validation loss:",
                checkpoint[
                    "best_validation_loss"
                ],
            )


@torch.no_grad()
def main():

    print("=" * 70)
    print("H2DiLR Stage-1 Diagnostic")
    print("=" * 70)

    device = Config.DEVICE

    print("Device:", device)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    print("\nLoading dataset...")

    _, val_loader = get_dataloaders()

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = create_model().to(device)

    checkpoint_path = (
        Path(Config.CHECKPOINT_DIR)
        / "h2d_stage1_best.pt"
    )

    load_checkpoint(
        model,
        checkpoint_path,
        device,
    )

    model.eval()

    # --------------------------------------------------
    # Code-usage counters
    # --------------------------------------------------

    shared_counts = torch.zeros(
        Config.SHARED_CODEBOOK_SIZE,
        dtype=torch.long,
    )

    private_counts = [
        torch.zeros(
            Config.PRIVATE_CODEBOOK_SIZE,
            dtype=torch.long,
        )
        for _ in range(
            Config.NUM_SUBJECTS
        )
    ]

    # --------------------------------------------------
    # Error statistics
    # --------------------------------------------------

    reconstruction_squared_error = 0.0
    zero_squared_error = 0.0
    quantization_squared_error = 0.0

    input_elements = 0
    latent_elements = 0

    shared_token_count = 0
    private_token_count = 0
    sample_count = 0

    latent_sum = 0.0
    latent_squared_sum = 0.0

    quantized_sum = 0.0
    quantized_squared_sum = 0.0

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    for batch_index, batch in enumerate(
        val_loader
    ):

        if (
            MAX_BATCHES is not None
            and batch_index >= MAX_BATCHES
        ):
            break

        emg, _, subject_ids = batch

        emg = emg.to(device)
        subject_ids = subject_ids.to(device)

        output = model(
            emg,
            subject_ids,
        )

        reconstruction = output[
            "reconstruction"
        ]

        latent = output[
            "latent_tokens"
        ]

        quantized = output[
            "quantized_tokens"
        ]

        # ----------------------------------------------
        # Reconstruction
        # ----------------------------------------------

        reconstruction_squared_error += (
            (
                reconstruction - emg
            ) ** 2
        ).sum().item()

        # Zero-output baseline:
        #
        # If normalized data have approximately
        # zero mean, predicting zero is a useful
        # trivial reconstruction reference.
        zero_squared_error += (
            emg ** 2
        ).sum().item()

        input_elements += emg.numel()

        # ----------------------------------------------
        # Quantization gap
        # ----------------------------------------------

        quantization_squared_error += (
            (
                latent - quantized
            ) ** 2
        ).sum().item()

        latent_elements += latent.numel()

        # ----------------------------------------------
        # Representation statistics
        # ----------------------------------------------

        latent_sum += (
            latent.sum().item()
        )

        latent_squared_sum += (
            (latent ** 2).sum().item()
        )

        quantized_sum += (
            quantized.sum().item()
        )

        quantized_squared_sum += (
            (quantized ** 2).sum().item()
        )

        # ----------------------------------------------
        # Shared code usage
        # ----------------------------------------------

        shared_mask = output[
            "shared_mask"
        ]

        shared_indices = output[
            "shared_indices"
        ]

        selected_shared_indices = (
            shared_indices[
                shared_mask
            ]
        )

        if (
            selected_shared_indices.numel()
            > 0
        ):

            counts = torch.bincount(
                selected_shared_indices.cpu(),
                minlength=(
                    Config.SHARED_CODEBOOK_SIZE
                ),
            )

            shared_counts += counts

        # ----------------------------------------------
        # Private code usage
        # ----------------------------------------------

        private_mask = output[
            "private_mask"
        ]

        private_indices = output[
            "private_indices"
        ]

        for subject_index in range(
            Config.NUM_SUBJECTS
        ):

            subject_mask = (
                subject_ids
                == subject_index
            )

            mask = (
                subject_mask.unsqueeze(1)
                & private_mask
            )

            subject_private_indices = (
                private_indices[mask]
            )

            if (
                subject_private_indices.numel()
                > 0
            ):

                counts = torch.bincount(
                    subject_private_indices.cpu(),
                    minlength=(
                        Config.PRIVATE_CODEBOOK_SIZE
                    ),
                )

                private_counts[
                    subject_index
                ] += counts

        shared_token_count += (
            shared_mask.sum().item()
        )

        private_token_count += (
            private_mask.sum().item()
        )

        sample_count += emg.shape[0]

    # --------------------------------------------------
    # Calculate results
    # --------------------------------------------------

    reconstruction_mse = (
        reconstruction_squared_error
        / input_elements
    )

    zero_baseline_mse = (
        zero_squared_error
        / input_elements
    )

    quantization_mse = (
        quantization_squared_error
        / latent_elements
    )

    reconstruction_improvement = (
        100.0
        * (
            zero_baseline_mse
            - reconstruction_mse
        )
        / zero_baseline_mse
    )

    latent_mean = (
        latent_sum
        / latent_elements
    )

    latent_variance = (
        latent_squared_sum
        / latent_elements
        - latent_mean ** 2
    )

    latent_std = max(
        latent_variance,
        0.0,
    ) ** 0.5

    quantized_mean = (
        quantized_sum
        / latent_elements
    )

    quantized_variance = (
        quantized_squared_sum
        / latent_elements
        - quantized_mean ** 2
    )

    quantized_std = max(
        quantized_variance,
        0.0,
    ) ** 0.5

    shared_used = (
        shared_counts > 0
    ).sum().item()

    shared_perplexity = (
        calculate_perplexity(
            shared_counts
        )
    )

    # --------------------------------------------------
    # Print report
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("RECONSTRUCTION")
    print("=" * 70)

    print(
        f"Zero-predictor MSE: "
        f"{zero_baseline_mse:.6f}"
    )

    print(
        f"Stage-1 reconstruction MSE: "
        f"{reconstruction_mse:.6f}"
    )

    print(
        f"Improvement over zero baseline: "
        f"{reconstruction_improvement:.2f}%"
    )

    print()
    print("=" * 70)
    print("REPRESENTATION")
    print("=" * 70)

    print(
        f"Latent mean: "
        f"{latent_mean:.6f}"
    )

    print(
        f"Latent std:  "
        f"{latent_std:.6f}"
    )

    print(
        f"Quantized mean: "
        f"{quantized_mean:.6f}"
    )

    print(
        f"Quantized std:  "
        f"{quantized_std:.6f}"
    )

    print(
        f"Latent → quantized MSE: "
        f"{quantization_mse:.6f}"
    )

    print()
    print("=" * 70)
    print("TOKEN ROUTING")
    print("=" * 70)

    print(
        "Samples evaluated:",
        sample_count,
    )

    print(
        "Average shared tokens/sample:",
        shared_token_count
        / sample_count,
    )

    print(
        "Average private tokens/sample:",
        private_token_count
        / sample_count,
    )

    print()
    print("=" * 70)
    print("SHARED CODEBOOK")
    print("=" * 70)

    print(
        "Shared codebook size:",
        Config.SHARED_CODEBOOK_SIZE,
    )

    print(
        "Shared codes used:",
        shared_used,
    )

    print(
        "Shared utilization:",
        f"{100.0 * shared_used / Config.SHARED_CODEBOOK_SIZE:.2f}%"
    )

    print(
        "Shared perplexity:",
        f"{shared_perplexity:.2f}"
    )

    print()
    print("=" * 70)
    print("PRIVATE CODEBOOKS")
    print("=" * 70)

    for subject_index in range(
        Config.NUM_SUBJECTS
    ):

        counts = private_counts[
            subject_index
        ]

        used = (
            counts > 0
        ).sum().item()

        perplexity = (
            calculate_perplexity(
                counts
            )
        )

        print(
            f"\nSubject {subject_index}"
        )

        print(
            "Private codebook size:",
            Config.PRIVATE_CODEBOOK_SIZE,
        )

        print(
            "Private codes used:",
            used,
        )

        print(
            "Private utilization:",
            f"{100.0 * used / Config.PRIVATE_CODEBOOK_SIZE:.2f}%"
        )

        print(
            "Private perplexity:",
            f"{perplexity:.2f}"
        )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
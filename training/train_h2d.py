"""
Training and validation utilities for H2DiLR Stage 1.
"""

from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float = 1.0,
) -> Dict[str, float]:
    """
    Train the Stage-1 H2DiLR model for one epoch.
    """

    model.train()

    running_total_loss = 0.0
    running_reconstruction_loss = 0.0
    running_quantization_loss = 0.0
    running_private_loss = 0.0
    running_commitment_loss = 0.0

    total_samples = 0

    progress_bar = tqdm(
        data_loader,
        desc="Training",
        leave=False,
    )

    for emg, _, subject_ids in progress_bar:
        emg = emg.to(
            device,
            non_blocking=True,
        )

        subject_ids = subject_ids.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        output = model(
            emg,
            subject_ids,
        )

        total_loss = output["total_loss"]

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Non-finite training loss detected: "
                f"{total_loss.item()}"
            )

        total_loss.backward()

        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )

        optimizer.step()

        batch_size = emg.shape[0]
        total_samples += batch_size

        running_total_loss += (
            output["total_loss"].item() * batch_size
        )

        running_reconstruction_loss += (
            output["reconstruction_loss"].item()
            * batch_size
        )

        running_quantization_loss += (
            output["quantization_loss"].item()
            * batch_size
        )

        running_private_loss += (
            output["private_codebook_loss"].item()
            * batch_size
        )

        running_commitment_loss += (
            output["commitment_loss"].item()
            * batch_size
        )

        progress_bar.set_postfix(
            total=f"{total_loss.item():.4f}",
            reconstruction=(
                f"{output['reconstruction_loss'].item():.4f}"
            ),
        )

    if total_samples == 0:
        raise RuntimeError(
            "Training DataLoader returned no samples."
        )

    return {
        "total_loss": (
            running_total_loss / total_samples
        ),
        "reconstruction_loss": (
            running_reconstruction_loss / total_samples
        ),
        "quantization_loss": (
            running_quantization_loss / total_samples
        ),
        "private_codebook_loss": (
            running_private_loss / total_samples
        ),
        "commitment_loss": (
            running_commitment_loss / total_samples
        ),
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the Stage-1 H2DiLR model for one epoch.
    """

    model.eval()

    running_total_loss = 0.0
    running_reconstruction_loss = 0.0
    running_quantization_loss = 0.0
    running_private_loss = 0.0
    running_commitment_loss = 0.0

    total_samples = 0

    progress_bar = tqdm(
        data_loader,
        desc="Validation",
        leave=False,
    )

    for emg, _, subject_ids in progress_bar:
        emg = emg.to(
            device,
            non_blocking=True,
        )

        subject_ids = subject_ids.to(
            device,
            non_blocking=True,
        )

        output = model(
            emg,
            subject_ids,
        )

        batch_size = emg.shape[0]
        total_samples += batch_size

        running_total_loss += (
            output["total_loss"].item() * batch_size
        )

        running_reconstruction_loss += (
            output["reconstruction_loss"].item()
            * batch_size
        )

        running_quantization_loss += (
            output["quantization_loss"].item()
            * batch_size
        )

        running_private_loss += (
            output["private_codebook_loss"].item()
            * batch_size
        )

        running_commitment_loss += (
            output["commitment_loss"].item()
            * batch_size
        )

    if total_samples == 0:
        raise RuntimeError(
            "Validation DataLoader returned no samples."
        )

    return {
        "total_loss": (
            running_total_loss / total_samples
        ),
        "reconstruction_loss": (
            running_reconstruction_loss / total_samples
        ),
        "quantization_loss": (
            running_quantization_loss / total_samples
        ),
        "private_codebook_loss": (
            running_private_loss / total_samples
        ),
        "commitment_loss": (
            running_commitment_loss / total_samples
        ),
    }


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_validation_loss: float,
    history: Dict[str, list],
) -> None:
    """
    Save a Stage-1 training checkpoint.
    """

    path = Path(checkpoint_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "best_validation_loss": (
            best_validation_loss
        ),
        "history": history,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = (
            scheduler.state_dict()
        )

    torch.save(
        checkpoint,
        path,
    )
"""
Evaluation utilities for Stage-2 H2DiLR.
"""

from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation.metrics import calculate_metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[
    Dict[str, float],
    np.ndarray,
    np.ndarray,
]:
    """
    Evaluate Stage-2 classifier.
    """

    model.eval()

    all_targets = []
    all_predictions = []

    running_loss = 0.0
    total_samples = 0

    for (
        emg,
        labels,
        subject_ids,
    ) in data_loader:

        emg = emg.to(device)

        labels = labels.to(device)

        subject_ids = subject_ids.to(
            device
        )

        output = model(
            emg,
            subject_ids,
        )

        logits = output[
            "logits"
        ]

        loss = (
            nn.functional.cross_entropy(
                logits,
                labels,
            )
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        batch_size = (
            labels.size(0)
        )

        running_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

        all_targets.append(
            labels.cpu().numpy()
        )

        all_predictions.append(
            predictions.cpu().numpy()
        )

    targets = np.concatenate(
        all_targets
    )

    predictions = np.concatenate(
        all_predictions
    )

    metrics = calculate_metrics(
        targets,
        predictions,
    )

    metrics["loss"] = (
        running_loss
        / total_samples
    )

    return (
        metrics,
        targets,
        predictions,
    )
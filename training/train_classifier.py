"""
Stage-2 training utilities for H2DiLR.
"""

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_classifier_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:

    model.train()

    # Keep frozen Stage-1 representation modules in eval mode.
    model.stage1.encoder.eval()
    model.stage1.quantizer.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        data_loader,
        desc="Stage-2 Training",
        leave=False,
    )

    for emg, labels, subject_ids in progress_bar:

        emg = emg.to(device)
        labels = labels.to(device)
        subject_ids = subject_ids.to(device)

        optimizer.zero_grad(set_to_none=True)

        output = model(
            emg,
            subject_ids,
        )

        logits = output["logits"]

        loss = nn.functional.cross_entropy(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{100.0 * correct / total:.2f}%",
        )

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def validate_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for emg, labels, subject_ids in data_loader:

        emg = emg.to(device)
        labels = labels.to(device)
        subject_ids = subject_ids.to(device)

        output = model(
            emg,
            subject_ids,
        )

        logits = output["logits"]

        loss = nn.functional.cross_entropy(
            logits,
            labels,
        )

        batch_size = labels.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }
"""
Training functions for the reconstruction-only control.
"""

import torch
from tqdm import tqdm


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    progress = tqdm(
        loader,
        desc="Training AE",
        leave=False,
    )

    for emg, _, _ in progress:

        emg = emg.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(emg)

        loss = output["total_loss"]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = emg.shape[0]

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / total_samples


@torch.no_grad()
def validate(
    model,
    loader,
    device,
):

    model.eval()

    reconstruction_error = 0.0
    zero_error = 0.0
    total_elements = 0

    for emg, _, _ in loader:

        emg = emg.to(device)

        output = model(emg)

        reconstruction = output[
            "reconstruction"
        ]

        reconstruction_error += (
            (
                reconstruction - emg
            ) ** 2
        ).sum().item()

        zero_error += (
            emg ** 2
        ).sum().item()

        total_elements += emg.numel()

    reconstruction_mse = (
        reconstruction_error
        / total_elements
    )

    zero_mse = (
        zero_error
        / total_elements
    )

    improvement = (
        100.0
        * (zero_mse - reconstruction_mse)
        / zero_mse
    )

    return (
        reconstruction_mse,
        zero_mse,
        improvement,
    )
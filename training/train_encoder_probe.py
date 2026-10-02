"""
Training utilities for the frozen encoder gesture probe.
"""

import torch


def train_probe_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:

        X, y_gesture, _ = batch

        X = X.to(device)
        y_gesture = y_gesture.to(device)

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(
            logits,
            y_gesture,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * X.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions == y_gesture
        ).sum().item()

        total_samples += X.size(0)

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        100.0
        * total_correct
        / total_samples
    )

    return average_loss, accuracy


@torch.no_grad()
def validate_probe(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:

        X, y_gesture, _ = batch

        X = X.to(device)
        y_gesture = y_gesture.to(device)

        logits = model(X)

        loss = criterion(
            logits,
            y_gesture,
        )

        total_loss += (
            loss.item() * X.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions == y_gesture
        ).sum().item()

        total_samples += X.size(0)

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        100.0
        * total_correct
        / total_samples
    )

    return average_loss, accuracy
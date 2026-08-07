"""
Classification metrics for H2DiLR Stage 2.
"""

from typing import Dict

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def calculate_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate multiclass classification metrics.
    """

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    precision = precision_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
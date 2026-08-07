"""
Confusion-matrix utilities for H2DiLR.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix


def save_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_path: str,
) -> None:
    """
    Save a confusion matrix as an image.
    """

    matrix = confusion_matrix(
        targets,
        predictions,
    )

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(14, 12)
    )

    plt.imshow(
        matrix,
        interpolation="nearest",
    )

    plt.title(
        "H2DiLR Gesture Classification Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()
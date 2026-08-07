"""
Data preprocessing for the NinaPro DB2 dataset.

Important:
Windows are created only inside a continuous gesture/repetition segment.
This prevents a window from containing samples from two different gestures.
"""

import os

import numpy as np
import scipy.io as sio

from configs.config import Config


def load_db2_dataset(
    data_path: str = Config.DATA_ROOT,
    window_size: int = Config.WINDOW_SIZE,
    step_size: int = Config.STEP_SIZE,
):
    """
    Load NinaPro DB2 EMG files and create clean sliding windows.

    Returns
    -------
    X : np.ndarray
        Shape:
        [num_windows, window_size, num_channels]

    y_gesture : np.ndarray
        Gesture labels.

    y_subject : np.ndarray
        Contiguous subject labels 0, 1, ...
    """

    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_path}"
        )

    X = []
    y_gesture = []
    y_subject = []

    files = sorted(
        filename
        for filename in os.listdir(data_path)
        if filename.lower().endswith(".mat")
    )

    if not files:
        raise FileNotFoundError(
            f"No .mat files found inside {data_path}"
        )

    print(f"Found {len(files)} files.")

    for filename in files:

        file_path = os.path.join(
            data_path,
            filename,
        )

        mat = sio.loadmat(file_path)

        if "emg" not in mat:
            raise KeyError(
                f"'emg' missing from {filename}"
            )

        if "stimulus" not in mat:
            raise KeyError(
                f"'stimulus' missing from {filename}"
            )

        if "repetition" not in mat:
            raise KeyError(
                f"'repetition' missing from {filename}"
            )

        emg = np.asarray(
            mat["emg"],
            dtype=np.float32,
        )

        stimulus = np.asarray(
            mat["stimulus"]
        ).reshape(-1)

        repetition = np.asarray(
            mat["repetition"]
        ).reshape(-1)

        if not (
            len(emg)
            == len(stimulus)
            == len(repetition)
        ):
            raise ValueError(
                f"Length mismatch in {filename}"
            )

        # ------------------------------------------------------
        # Extract subject ID
        # ------------------------------------------------------

        # Example:
        # S7_E1_A1.mat -> 6
        subject_prefix = filename.split("_")[0]

        original_subject_id = (
            int(subject_prefix[1:]) - 1
        )

        # ------------------------------------------------------
        # Find continuous gesture/repetition segments
        # ------------------------------------------------------

        segment_start = 0

        for index in range(1, len(stimulus) + 1):

            end_of_signal = (
                index == len(stimulus)
            )

            if end_of_signal:
                segment_changed = True

            else:
                stimulus_changed = (
                    stimulus[index]
                    != stimulus[index - 1]
                )

                repetition_changed = (
                    repetition[index]
                    != repetition[index - 1]
                )

                segment_changed = (
                    stimulus_changed
                    or repetition_changed
                )

            if not segment_changed:
                continue

            segment_end = index

            current_gesture = int(
                stimulus[segment_start]
            )

            current_repetition = int(
                repetition[segment_start]
            )

            # --------------------------------------------------
            # Ignore rest state
            # --------------------------------------------------

            if current_gesture > 0:

                segment_emg = emg[
                    segment_start:segment_end
                ]

                segment_length = len(
                    segment_emg
                )

                # ----------------------------------------------
                # Create windows only WITHIN this segment
                # ----------------------------------------------

                if segment_length >= window_size:

                    for start in range(
                        0,
                        segment_length
                        - window_size
                        + 1,
                        step_size,
                    ):

                        stop = (
                            start + window_size
                        )

                        window = segment_emg[
                            start:stop
                        ]

                        X.append(window)

                        y_gesture.append(
                            current_gesture - 1
                        )

                        y_subject.append(
                            original_subject_id
                        )

            segment_start = segment_end

    if not X:
        raise RuntimeError(
            "No windows were generated."
        )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y_gesture = np.asarray(
        y_gesture,
        dtype=np.int64,
    )

    y_subject = np.asarray(
        y_subject,
        dtype=np.int64,
    )

    # ----------------------------------------------------------
    # Remap subject IDs
    #
    # Example:
    # 6  -> 0
    # 10 -> 1
    # ----------------------------------------------------------

    unique_subjects = np.unique(
        y_subject
    )

    subject_mapping = {
        int(original_id): new_id
        for new_id, original_id
        in enumerate(unique_subjects)
    }

    y_subject = np.asarray(
        [
            subject_mapping[
                int(subject_id)
            ]
            for subject_id
            in y_subject
        ],
        dtype=np.int64,
    )

    print(
        "Dataset Loaded Successfully"
    )

    print(
        "EMG Windows :",
        X.shape,
    )

    print(
        "Gesture Labels :",
        y_gesture.shape,
    )

    print(
        "Subjects :",
        y_subject.shape,
    )

    print(
        "Subject mapping:",
        subject_mapping,
    )

    print(
        "Gesture classes:",
        len(np.unique(y_gesture)),
    )

    print(
        "Gesture range:",
        y_gesture.min(),
        "to",
        y_gesture.max(),
    )

    return (
        X,
        y_gesture,
        y_subject,
    )


def normalize_emg(
    X: np.ndarray,
) -> np.ndarray:
    """
    Channel-wise z-score normalization.
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    if X.ndim != 3:
        raise ValueError(
            "Expected X shape "
            "[windows, time, channels]."
        )

    mean = X.mean(
        axis=(0, 1),
        keepdims=True,
    )

    std = X.std(
        axis=(0, 1),
        keepdims=True,
    )

    std = np.maximum(
        std,
        1e-8,
    )

    X = (
        X - mean
    ) / std

    return X.astype(
        np.float32,
        copy=False,
    )
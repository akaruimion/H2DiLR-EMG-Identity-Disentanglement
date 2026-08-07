"""
PyTorch Dataset for NinaPro DB2
"""

import torch
from torch.utils.data import Dataset


class EMGDataset(Dataset):
    """
    PyTorch Dataset for EMG windows.
    """

    def __init__(self, X, y_gesture, y_subject=None):
        """
        Parameters
        ----------
        X : numpy.ndarray
            EMG windows
        y_gesture : numpy.ndarray
            Gesture labels
        y_subject : numpy.ndarray, optional
            Subject labels
        """

        self.X = torch.FloatTensor(X)
        self.y_gesture = torch.LongTensor(y_gesture)

        if y_subject is not None:
            self.y_subject = torch.LongTensor(y_subject)
        else:
            self.y_subject = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        if self.y_subject is None:
            return self.X[idx], self.y_gesture[idx]

        return (
            self.X[idx],
            self.y_gesture[idx],
            self.y_subject[idx]
        )
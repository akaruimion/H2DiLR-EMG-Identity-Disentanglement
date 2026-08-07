"""
PyTorch DataLoaders for NinaPro DB2
"""

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from configs.config import Config
from datasets.preprocess import load_db2_dataset, normalize_emg
from datasets.dataset import EMGDataset


def get_dataloaders():
    """
    Returns:
        train_loader
        val_loader
    """

    # Load dataset
    X, y_gesture, y_subject = load_db2_dataset()

    # Normalize
    X = normalize_emg(X)

    # Train/Validation Split
    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
        X,
        y_gesture,
        y_subject,
        test_size=0.2,
        random_state=Config.SEED,
        stratify=y_gesture,
    )

    # Dataset objects
    train_dataset = EMGDataset(X_train, y_train, s_train)
    val_dataset = EMGDataset(X_val, y_val, s_val)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # We'll change this later if needed
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader
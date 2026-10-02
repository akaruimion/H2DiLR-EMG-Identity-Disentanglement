"""
Configuration for the H2DiLR project.
"""

import os
import torch


class Config:

    # ==========================================================
    # Dataset
    # ==========================================================

    DATA_ROOT = "data"

    NUM_CHANNELS = 12
    NUM_CLASSES = 49
    NUM_SUBJECTS = 2

    # ==========================================================
    # Windowing
    # ==========================================================

    WINDOW_SIZE = 200
    STEP_SIZE = 50

    # ==========================================================
    # DataLoader
    # ==========================================================

    BATCH_SIZE = 64
    NUM_WORKERS = 0

    # ==========================================================
    # Stage-1 Pretraining
    # ==========================================================

    PRETRAIN_EPOCHS = 10

    PRETRAIN_LEARNING_RATE = 5e-5
    PRIVATE_CODEBOOK_LEARNING_RATE = 2e-4

    WEIGHT_DECAY = 0.01
    GRADIENT_CLIP_NORM = 1.0

    # ==========================================================
    # Encoder
    # ==========================================================

    ENCODER_INPUT_CHANNELS = 12
    ENCODER_STEM_CHANNELS = 64
    LATENT_DIM = 256

    # ==========================================================
    # Standard Vector Quantization
    # ==========================================================

    CODEBOOK_SIZE = 256
    COMMITMENT_BETA = 0.25

    # ==========================================================
    # H2D Quantization
    # ==========================================================

    SHARED_RATIO = 0.5

    PRIVATE_CODEBOOK_SIZE = 64

    SHARED_CODEBOOK_SIZE = (
        NUM_SUBJECTS * PRIVATE_CODEBOOK_SIZE
    )

    EMA_DECAY = 0.90
    EMA_EPSILON = 1e-5

    # ==========================================================
    # Stage-2 Transformer
    # ==========================================================

    TRANSFORMER_DIM = 128
    TRANSFORMER_HEADS = 4
    TRANSFORMER_LAYERS = 4
    TRANSFORMER_FFN_DIM = 512
    TRANSFORMER_DROPOUT = 0.1

    # ==========================================================
    # Stage-2 Training
    # ==========================================================

    FINETUNE_EPOCHS = 15
    FINETUNE_LEARNING_RATE = 5e-5

    # ==========================================================
    # Checkpoints
    # ==========================================================

    STAGE1_CHECKPOINT = "checkpoints/h2d_stage1_best.pt"
    STAGE2_CHECKPOINT = "checkpoints/h2d_stage2_best.pt"

    # ==========================================================
    # Device
    # ==========================================================

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ==========================================================
    # Reproducibility
    # ==========================================================

    SEED = 42

    # ==========================================================
    # Output directories
    # ==========================================================

    CHECKPOINT_DIR = "checkpoints"
    RESULT_DIR = "results"


# ==============================================================
# Create output directories automatically
# ==============================================================

os.makedirs(
    Config.CHECKPOINT_DIR,
    exist_ok=True,
)

os.makedirs(
    Config.RESULT_DIR,
    exist_ok=True,
)
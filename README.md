# H2DiLR for NinaPro DB2 EMG Gesture Recognition

## Overview

This repository contains a PyTorch implementation and adaptation of **H2DiLR (Homogeneity-Heterogeneity Disentangled Learning for Neural Representations)** for the **NinaPro DB2 surface EMG dataset**.

The original H2DiLR framework was proposed for learning shared and subject-specific representations from heterogeneous neural recordings. In this project, the same core idea is adapted to multi-subject EMG gesture recognition.

The main goal is to investigate whether the learned representation can:

- preserve gesture-related information,
- separate shared information from subject-specific information,
- reduce identity information in the shared representation,
- and support downstream gesture classification.

> **Status:** The core end-to-end implementation is complete and running. Current results are preliminary development results. Identity-probe experiments, stronger evaluation protocols, baselines, ablations, and larger multi-subject experiments are still in progress.

## Research Objective

EMG signals vary significantly between subjects even when they perform the same gesture. The project therefore studies whether H2D-style representation learning can disentangle:

- **Shared / homogeneous information** — patterns useful across subjects and potentially related to gesture content.
- **Private / heterogeneous information** — patterns specific to individual subjects.

The current pipeline follows two stages:

1. **Stage 1 — H2D self-supervised representation learning**
2. **Stage 2 — Supervised gesture recognition using a Transformer classifier**

## Dataset

The project uses MATLAB `.mat` files from the **NinaPro DB2** dataset.

Current development subset:

- 2 subjects
- 6 `.mat` files
- 12 EMG channels
- 49 gesture classes
- gesture labels represented internally as `0–48`

Example data files:

```text
data/
├── S7_E1_A1.mat
├── S7_E2_A1.mat
├── S7_E3_A1.mat
├── S11_E1_A1.mat
├── S11_E2_A1.mat
└── S11_E3_A1.mat
```

The raw dataset files are **not included in this repository**.

## Preprocessing

The preprocessing pipeline:

1. Loads `emg`, `stimulus`, and `repetition` information from each `.mat` file.
2. Identifies continuous gesture/repetition segments.
3. Excludes rest segments.
4. Creates fixed-length EMG windows only within valid continuous segments.
5. Assigns one gesture label and one subject label to every window.
6. Remaps original subject identifiers to contiguous internal subject indices.

Current preprocessing output:

```text
EMG Windows   : (125667, 200, 12)
Gesture Labels: (125667,)
Subjects      : (125667,)

Gesture classes: 49
Gesture range  : 0 to 48
```

Each EMG sample has shape:

```text
[200, 12]
```

where `200` is the number of temporal samples and `12` is the number of EMG channels.

Current sliding-window configuration:

```text
Window size = 200
Step size   = 50
```

## Project Architecture

```text
NinaPro DB2 .mat files
        │
        ▼
   Preprocessing
        │
        ▼
EMG windows [B, 200, 12]
        │
        ▼
────────────────────────────────
        Stage 1: H2D
────────────────────────────────
        │
        ▼
     Encoder
        │
        ▼
Latent tokens [B, 10, 256]
        │
        ▼
   H2D Quantizer
     ↙         ↘
 Shared       Private
Codebook     Codebooks
     ↘         ↙
 Quantized H2D tokens
        │
        ▼
      Decoder
        │
        ▼
Reconstructed EMG [B, 200, 12]

────────────────────────────────
        Stage 2
────────────────────────────────
        │
        ▼
Frozen Stage-1 Representation
        │
        ▼
H2D tokens [B, 10, 256]
        │
        ▼
    Transformer
        │
        ▼
Representation [B, 128]
        │
        ▼
     Classifier
        │
        ▼
Gesture logits [B, 49]
```

## Stage 1 — H2D Representation Learning

Stage 1 learns a quantized representation of the EMG signal without using gesture labels for the reconstruction objective.

### Encoder

The 1-D convolutional encoder transforms:

```text
[B, 200, 12] → [B, 10, 256]
```

Each sample is represented by 10 latent tokens with 256 features per token.

### Vector Quantization

A standard vector quantizer is implemented for testing and as a foundation for H2D quantization. The implementation includes:

- nearest-codebook lookup,
- codebook loss,
- commitment loss,
- straight-through estimation,
- codebook usage/perplexity measurements.

### H2D Quantizer

The H2D quantizer separates latent tokens into **shared tokens** and **subject-private tokens**.

Current setting:

```text
Shared ratio ν = 0.5
```

For 10 latent tokens:

```text
5 tokens → shared codebook
5 tokens → private codebook
```

The shared codebook is common across subjects, while each subject has its own private codebook.

Current commitment setting:

```text
β = 0.25
```

### Decoder

The decoder reconstructs the original EMG shape:

```text
[B, 10, 256] → [B, 200, 12]
```

Stage-1 training uses reconstruction and quantization-related losses.

## Stage 2 — Gesture Classification

After Stage-1 pretraining, the representation-learning modules are frozen.

The H2D tokens are passed to a Transformer-based downstream classifier:

```text
H2D tokens [B, 10, 256]
        ↓
Transformer
        ↓
[B, 128]
        ↓
Classifier
        ↓
[B, 49]
```

Only the Transformer and gesture classifier are optimized during Stage 2.

## Project Structure

```text
H2DiLRProject/
│
├── configs/
│   ├── __init__.py
│   └── config.py
│
├── datasets/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── dataset.py
│   ├── dataloader.py
│   └── windowing.py
│
├── models/
│   ├── __init__.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── vector_quantizer.py
│   ├── h2d_quantizer.py
│   ├── h2dilr.py
│   ├── transformer.py
│   ├── classifier.py
│   └── h2d_stage2.py
│
├── training/
│   ├── __init__.py
│   ├── train_h2d.py
│   └── train_classifier.py
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   ├── evaluate.py
│   └── confusion_matrix.py
│
├── results/
│   └── confusion_matrix.png
│
├── pretrain.py
├── finetune.py
├── evaluate.py
├── test.py
├── requirements.txt
├── .gitignore
└── README.md
```

## File Descriptions

### `configs/config.py`
Stores dataset, model, training, Transformer, codebook, checkpoint, and device configuration.

### `datasets/preprocess.py`
Loads NinaPro DB2 `.mat` files, extracts EMG/gesture/repetition information, creates valid segments, generates windows, assigns labels, and remaps subject IDs.

### `datasets/windowing.py`
Implements sliding-window segmentation using the configured window and step sizes.

### `datasets/dataset.py`
Implements the PyTorch `Dataset` interface and returns an EMG window, gesture label, and subject label.

### `datasets/dataloader.py`
Creates training and validation DataLoaders and groups samples into mini-batches.

### `models/encoder.py`
Implements the 1-D convolutional encoder:

```text
[B, 200, 12] → [B, 10, 256]
```

### `models/vector_quantizer.py`
Implements standard vector quantization, including nearest-code selection, codebook loss, commitment loss, and straight-through gradient estimation.

### `models/h2d_quantizer.py`
Implements H2D shared/private quantization using one shared codebook, one private codebook per subject, token routing, private-codebook optimization, and shared-codebook EMA updating.

### `models/decoder.py`
Reconstructs EMG from quantized latent tokens:

```text
[B, 10, 256] → [B, 200, 12]
```

### `models/h2dilr.py`
Combines `Encoder → H2D Quantizer → Decoder` into the complete Stage-1 model and provides representation-freezing functionality.

### `training/train_h2d.py`
Contains Stage-1 training and validation loops, loss tracking, gradient clipping, and checkpoint logic.

### `pretrain.py`
Main Stage-1 training entry point.

### `models/transformer.py`
Processes the H2D token sequence and creates a compact downstream representation:

```text
[B, 10, 256] → [B, 128]
```

### `models/classifier.py`
Maps the Transformer representation to 49 gesture logits:

```text
[B, 128] → [B, 49]
```

### `models/h2d_stage2.py`
Combines the frozen Stage-1 representation model with the Transformer and gesture classifier.

### `training/train_classifier.py`
Contains Stage-2 supervised training and validation loops using cross-entropy loss.

### `finetune.py`
Main Stage-2 training entry point.

### `evaluation/metrics.py`
Computes accuracy, macro precision, macro recall, and macro F1 score.

### `evaluation/confusion_matrix.py`
Generates and saves the gesture confusion matrix.

### `evaluation/evaluate.py`
Contains reusable model-evaluation logic.

### `evaluate.py`
Loads trained checkpoints and reports evaluation metrics.

### `test.py`
Integration and sanity-test script used to verify tensor dimensions, forward passes, quantizer behavior, reconstruction, gradient propagation, Stage-1 freezing, Transformer output, and classifier output.

## Installation

Create and activate a Python virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dependencies

```text
torch
numpy
scipy
scikit-learn
matplotlib
tqdm
```

## Running the Project

### 1. Integration Test

```bash
python test.py
```

Typical tensor shapes:

```text
Input EMG              : [64, 200, 12]
Latent representation : [64, 10, 256]
Quantized tokens       : [64, 10, 256]
Reconstructed EMG      : [64, 200, 12]
Transformer output     : [64, 128]
Classifier logits      : [64, 49]
```

### 2. Stage-1 Pretraining

```bash
python pretrain.py
```

Best Stage-1 checkpoint:

```text
checkpoints/h2d_stage1_best.pt
```

### 3. Stage-2 Gesture Training

```bash
python finetune.py
```

Best Stage-2 checkpoint:

```text
checkpoints/h2d_stage2_best.pt
```

### 4. Evaluation

```bash
python evaluate.py
```

The script reports loss, accuracy, precision, recall, F1, and generates a confusion matrix.

## Preliminary Development Results

The following values are from a short development experiment and should **not** be interpreted as final benchmark performance:

```text
Validation Loss : 3.500335
Accuracy        : 8.22%
Precision       : 7.43%
Recall          : 8.30%
Macro F1        : 6.49%
```

With 49 gesture classes, uniform random chance is approximately `1/49 ≈ 2.04%`. The model is learning some gesture-related structure, but performance remains low and requires further training and experimental refinement.

## Current Limitations

- Only two subjects are currently used.
- Current validation results are preliminary.
- The current development split uses window-level sampling and should be replaced by a stricter repetition- or subject-aware protocol for final evaluation.
- Final normalization should be fitted using training data only.
- Full identity/disentanglement evaluation has not yet been completed.
- More extensive Stage-1 and Stage-2 training is required.
- Baseline, ablation, and repeated-seed experiments are still pending.

## Planned Identity-Disentanglement Evaluation

A key remaining experiment is an **identity probe**. A subject classifier will be trained on frozen representations to measure how much subject identity remains in:

```text
Raw EMG
Encoder latent representation
Shared H2D representation
Private H2D representation
Combined H2D representation
```

Desired behavior:

```text
Shared representation:
    high gesture information
    low subject-identity information

Private representation:
    stronger subject-specific information
```

## Planned Experiments

- longer Stage-1 pretraining,
- longer Stage-2 gesture training,
- identity-classification probes,
- repetition-based evaluation,
- subject-aware evaluation,
- additional NinaPro DB2 subjects,
- raw-EMG baseline,
- encoder-only baseline,
- shared-only H2D evaluation,
- private-only H2D evaluation,
- combined H2D evaluation,
- shared-ratio (`ν`) ablation,
- multiple random seeds,
- final statistical analysis and plots.

## Relationship to H2DiLR

This project is an **adaptation**, not an exact reproduction of the original H2DiLR experiment.

The main methodological components retained are:

- two-stage representation learning and decoding,
- 1-D convolutional feature extraction,
- vector quantization,
- shared and subject-private codebooks,
- homogeneity/heterogeneity token partitioning,
- reconstruction-based Stage-1 training,
- freezing the representation model before downstream training,
- Transformer-based downstream decoding.

Main adaptation:

```text
Original H2DiLR       → This project
-----------------------------------------------
sEEG recordings       → surface EMG
lexical tones         → hand gestures
brain electrodes      → 12 EMG channels
tone classifier       → 49-gesture classifier
```

## Reproducibility Notes

The following are intentionally excluded from GitHub:

```text
data/
checkpoints/
.venv/
*.mat
*.pt
```

Users must obtain NinaPro DB2 separately and place the required `.mat` files inside the `data/` directory.

## Project Status

**Implemented:** data loading, preprocessing, windowing, subject remapping, Dataset/DataLoader, encoder, vector quantizer, H2D shared/private quantizer, decoder, Stage-1 training, checkpointing, Transformer, 49-class classifier, Stage-2 training, metrics, confusion matrix, and integration testing.

**Remaining:** converged large-scale training, identity probes, stronger experimental splits, additional subjects, baselines, ablations, repeated-seed evaluation, and final scientific conclusions.

## Reference

This implementation is inspired by the H2DiLR framework:

**Towards Homogeneous Lexical Tone Decoding from Heterogeneous Intracranial Recordings**, ICLR 2025.

This repository adapts the H2DiLR representation-learning concept to NinaPro DB2 EMG gesture recognition and is **not** the official implementation of the original paper.

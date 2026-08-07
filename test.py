import torch

from configs.config import Config
from datasets.dataloader import get_dataloaders

from models.h2dilr import H2DiLRStage1
from models.h2d_stage2 import H2DiLRStage2


def create_stage1():

    model = H2DiLRStage1(
        input_channels=Config.NUM_CHANNELS,
        latent_dim=Config.LATENT_DIM,
        output_channels=Config.NUM_CHANNELS,
        output_length=Config.WINDOW_SIZE,
        num_subjects=Config.NUM_SUBJECTS,
        shared_codebook_size=(
            Config.SHARED_CODEBOOK_SIZE
        ),
        private_codebook_size=(
            Config.PRIVATE_CODEBOOK_SIZE
        ),
        shared_ratio=Config.SHARED_RATIO,
        commitment_beta=Config.COMMITMENT_BETA,
        ema_decay=Config.EMA_DECAY,
        ema_epsilon=Config.EMA_EPSILON,
    )

    return model


def main():

    train_loader, _ = get_dataloaders()

    stage1 = create_stage1()

    checkpoint = torch.load(
        "checkpoints/h2d_stage1_best.pt",
        map_location=Config.DEVICE,
    )

    stage1.load_state_dict(
        checkpoint["model_state_dict"]
    )

    stage1 = stage1.to(Config.DEVICE)

    model = H2DiLRStage2(
        stage1_model=stage1,
        latent_dim=Config.LATENT_DIM,
        transformer_dim=Config.TRANSFORMER_DIM,
        transformer_heads=Config.TRANSFORMER_HEADS,
        transformer_layers=Config.TRANSFORMER_LAYERS,
        transformer_ffn_dim=(
            Config.TRANSFORMER_FFN_DIM
        ),
        transformer_dropout=(
            Config.TRANSFORMER_DROPOUT
        ),
        num_classes=Config.NUM_CLASSES,
    ).to(Config.DEVICE)

    X, labels, subject_ids = next(
        iter(train_loader)
    )

    X = X.to(Config.DEVICE)
    labels = labels.to(Config.DEVICE)
    subject_ids = subject_ids.to(Config.DEVICE)

    output = model(
        X,
        subject_ids,
    )

    logits = output["logits"]

    print(
        "Input shape:",
        X.shape,
    )

    print(
        "H2D token shape:",
        output["h2d_tokens"].shape,
    )

    print(
        "Transformer representation:",
        output["representation"].shape,
    )

    print(
        "Classifier logits:",
        logits.shape,
    )

    loss = torch.nn.functional.cross_entropy(
        logits,
        labels,
    )

    print(
        "Classification loss:",
        loss.item(),
    )

    loss.backward()

    # Stage-1 must remain frozen.
    encoder_gradients = [
        parameter.grad
        for parameter in model.stage1.encoder.parameters()
    ]

    assert all(
        gradient is None
        for gradient in encoder_gradients
    )

    # Transformer should receive gradients.
    assert (
        model.transformer
        .input_projection
        .weight
        .grad
        is not None
    )

    # Classifier should receive gradients.
    assert (
        model.classifier
        .classifier[1]
        .weight
        .grad
        is not None
    )

    print(
        "Frozen Stage-1 test: passed"
    )

    print(
        "Transformer gradient test: passed"
    )

    print(
        "Classifier gradient test: passed"
    )

    print(
        "Stage-2 model test passed successfully."
    )


if __name__ == "__main__":
    main()
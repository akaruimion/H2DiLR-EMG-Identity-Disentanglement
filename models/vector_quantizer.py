"""
Standard vector quantization module for H2DiLR.

Input shape:
    [batch_size, num_tokens, embedding_dim]

Output shape:
    [batch_size, num_tokens, embedding_dim]
"""

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Standard learnable vector quantizer.

    Each continuous latent token is replaced by its nearest
    codebook embedding using squared Euclidean distance.

    Parameters
    ----------
    num_embeddings:
        Number of discrete vectors in the codebook.

    embedding_dim:
        Dimension of every codebook vector.

    commitment_beta:
        Weight applied to the commitment loss.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_beta: float = 0.25,
    ) -> None:
        super().__init__()

        if num_embeddings <= 0:
            raise ValueError(
                "num_embeddings must be greater than zero."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        if commitment_beta < 0:
            raise ValueError(
                "commitment_beta cannot be negative."
            )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_beta = commitment_beta

        self.codebook = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

        # Common VQ-VAE initialization.
        initialization_range = 1.0 / num_embeddings

        nn.init.uniform_(
            self.codebook.weight,
            -initialization_range,
            initialization_range,
        )

    def compute_distances(
        self,
        flat_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute squared Euclidean distances between latent tokens
        and every codebook vector.

        Parameters
        ----------
        flat_tokens:
            Tensor shaped [number_of_tokens, embedding_dim].

        Returns
        -------
        torch.Tensor
            Distance matrix shaped
            [number_of_tokens, num_embeddings].
        """

        codebook_vectors = self.codebook.weight

        # ||z - e||² = ||z||² + ||e||² - 2zeᵀ
        token_norm = torch.sum(
            flat_tokens ** 2,
            dim=1,
            keepdim=True,
        )

        codebook_norm = torch.sum(
            codebook_vectors ** 2,
            dim=1,
        ).unsqueeze(0)

        distances = (
            token_norm
            + codebook_norm
            - 2.0 * torch.matmul(
                flat_tokens,
                codebook_vectors.t(),
            )
        )

        return distances

    def forward(
        self,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize continuous latent tokens.

        Parameters
        ----------
        z:
            Continuous tokens with shape:

                [batch_size, num_tokens, embedding_dim]

        Returns
        -------
        dict
            Contains:

            quantized:
                Straight-through quantized representation.

            quantized_raw:
                Quantized embedding without straight-through estimator.

            indices:
                Selected codebook index for every token.

            distances:
                Distance from every token to every code.

            codebook_loss:
                Moves codebook vectors toward encoder outputs.

            commitment_loss:
                Encourages encoder outputs to remain close to codes.

            loss:
                codebook_loss + beta * commitment_loss.

            perplexity:
                Approximate codebook utilization measurement.
        """

        if z.ndim != 3:
            raise ValueError(
                "VectorQuantizer input must have shape "
                "[batch_size, num_tokens, embedding_dim]. "
                f"Received shape: {tuple(z.shape)}"
            )

        if z.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"but received {z.shape[-1]}."
            )

        original_shape = z.shape

        flat_tokens = z.reshape(
            -1,
            self.embedding_dim,
        )

        distances = self.compute_distances(flat_tokens)

        # Nearest-neighbour code index.
        encoding_indices = torch.argmin(
            distances,
            dim=1,
        )

        quantized_raw = self.codebook(
            encoding_indices
        ).view(original_shape)

        # Codebook loss:
        # codebook vectors move toward detached encoder outputs.
        codebook_loss = F.mse_loss(
            quantized_raw,
            z.detach(),
        )

        # Commitment loss:
        # encoder outputs move toward detached codebook vectors.
        commitment_loss = F.mse_loss(
            z,
            quantized_raw.detach(),
        )

        total_loss = (
            codebook_loss
            + self.commitment_beta * commitment_loss
        )

        # Straight-through estimator:
        # forward value is quantized_raw, but encoder receives gradients.
        quantized = z + (
            quantized_raw - z
        ).detach()

        indices = encoding_indices.view(
            original_shape[0],
            original_shape[1],
        )

        perplexity = self._calculate_perplexity(
            encoding_indices
        )

        return {
            "quantized": quantized,
            "quantized_raw": quantized_raw,
            "indices": indices,
            "distances": distances,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "loss": total_loss,
            "perplexity": perplexity,
        }

    def _calculate_perplexity(
        self,
        encoding_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate how many codebook entries are actively being used.
        """

        one_hot_encodings = F.one_hot(
            encoding_indices,
            num_classes=self.num_embeddings,
        ).float()

        average_probabilities = one_hot_encodings.mean(dim=0)

        perplexity = torch.exp(
            -torch.sum(
                average_probabilities
                * torch.log(
                    average_probabilities + 1e-10
                )
            )
        )

        return perplexity
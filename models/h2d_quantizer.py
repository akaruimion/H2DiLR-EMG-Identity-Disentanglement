"""
Homogeneity-Heterogeneity Disentangled (H2D) quantizer.

The module contains:

1. One shared codebook for homogeneous representations.
2. One private codebook for each subject.
3. Token routing based on distance to the shared codebook.
4. EMA updating for the shared codebook.
5. Gradient-based learning for private codebooks.

Input
-----
z:
    [batch_size, num_tokens, embedding_dim]

subject_ids:
    [batch_size]

Output
------
quantized:
    [batch_size, num_tokens, embedding_dim]
"""

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class H2DQuantizer(nn.Module):
    """
    Shared/private vector quantizer used during H2D pretraining.

    Parameters
    ----------
    num_subjects:
        Number of subjects in the dataset.

    shared_codebook_size:
        Number of entries in the shared codebook.

    private_codebook_size:
        Number of entries in each subject-private codebook.

    embedding_dim:
        Dimension of each latent token and codebook entry.

    shared_ratio:
        Fraction of tokens routed to the shared codebook.

    commitment_beta:
        Weight of the commitment loss.

    ema_decay:
        Momentum used for shared-codebook EMA updates.

    ema_epsilon:
        Small value used for numerical stability.
    """

    def __init__(
        self,
        num_subjects: int,
        shared_codebook_size: int,
        private_codebook_size: int,
        embedding_dim: int,
        shared_ratio: float = 0.5,
        commitment_beta: float = 0.25,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()

        if num_subjects <= 0:
            raise ValueError(
                "num_subjects must be greater than zero."
            )

        if shared_codebook_size <= 0:
            raise ValueError(
                "shared_codebook_size must be greater than zero."
            )

        if private_codebook_size <= 0:
            raise ValueError(
                "private_codebook_size must be greater than zero."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        if not 0.0 <= shared_ratio <= 1.0:
            raise ValueError(
                "shared_ratio must be between 0 and 1."
            )

        if commitment_beta < 0:
            raise ValueError(
                "commitment_beta cannot be negative."
            )

        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(
                "ema_decay must be between 0 and 1."
            )

        self.num_subjects = num_subjects
        self.shared_codebook_size = shared_codebook_size
        self.private_codebook_size = private_codebook_size
        self.embedding_dim = embedding_dim
        self.shared_ratio = shared_ratio
        self.commitment_beta = commitment_beta
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon

        # ------------------------------------------------------
        # Shared codebook
        # ------------------------------------------------------
        #
        # The shared codebook is updated using EMA rather than
        # ordinary gradient descent.
        self.shared_codebook = nn.Embedding(
            num_embeddings=shared_codebook_size,
            embedding_dim=embedding_dim,
        )

        shared_initialization_range = (
            1.0 / shared_codebook_size
        )

        nn.init.uniform_(
            self.shared_codebook.weight,
            -shared_initialization_range,
            shared_initialization_range,
        )

        self.shared_codebook.weight.requires_grad_(False)

        # ------------------------------------------------------
        # Private codebooks
        # ------------------------------------------------------
        #
        # Each subject receives a separate learnable codebook.
        self.private_codebooks = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=private_codebook_size,
                    embedding_dim=embedding_dim,
                )
                for _ in range(num_subjects)
            ]
        )

        private_initialization_range = (
            1.0 / private_codebook_size
        )

        for codebook in self.private_codebooks:
            nn.init.uniform_(
                codebook.weight,
                -private_initialization_range,
                private_initialization_range,
            )

    @staticmethod
    def _compute_distances(
        tokens: torch.Tensor,
        codebook_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate squared Euclidean distances.

        Parameters
        ----------
        tokens:
            [number_of_tokens, embedding_dim]

        codebook_vectors:
            [codebook_size, embedding_dim]

        Returns
        -------
        torch.Tensor
            [number_of_tokens, codebook_size]
        """

        token_norm = torch.sum(
            tokens ** 2,
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
                tokens,
                codebook_vectors.t(),
            )
        )

        return distances

    def _select_shared_mask(
        self,
        minimum_shared_distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select the tokens most similar to the shared codebook.

        Parameters
        ----------
        minimum_shared_distances:
            [batch_size, num_tokens]

        Returns
        -------
        torch.Tensor
            Boolean mask shaped [batch_size, num_tokens].
        """

        batch_size, num_tokens = (
            minimum_shared_distances.shape
        )

        number_of_shared_tokens = int(
            round(num_tokens * self.shared_ratio)
        )

        number_of_shared_tokens = max(
            0,
            min(number_of_shared_tokens, num_tokens),
        )

        shared_mask = torch.zeros(
            batch_size,
            num_tokens,
            dtype=torch.bool,
            device=minimum_shared_distances.device,
        )

        if number_of_shared_tokens == 0:
            return shared_mask

        shared_positions = torch.topk(
            minimum_shared_distances,
            k=number_of_shared_tokens,
            dim=1,
            largest=False,
        ).indices

        shared_mask.scatter_(
            dim=1,
            index=shared_positions,
            value=True,
        )

        return shared_mask

    @torch.no_grad()
    def _update_shared_codebook(
        self,
        selected_tokens: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> None:
        """
        Update used shared codes through exponential moving average.

        Only shared-code entries selected during the current batch
        are updated.
        """

        if selected_tokens.numel() == 0:
            return

        unique_indices = torch.unique(
            selected_indices
        )

        for code_index in unique_indices:
            code_mask = selected_indices == code_index

            assigned_tokens = selected_tokens[code_mask]

            if assigned_tokens.numel() == 0:
                continue

            batch_mean = assigned_tokens.mean(dim=0)

            code_number = int(code_index.item())

            old_code = self.shared_codebook.weight[
                code_number
            ]

            updated_code = (
                self.ema_decay * old_code
                + (1.0 - self.ema_decay) * batch_mean
            )

            self.shared_codebook.weight[
                code_number
            ].copy_(updated_code)

    @staticmethod
    def _calculate_perplexity(
        indices: torch.Tensor,
        codebook_size: int,
    ) -> torch.Tensor:
        """
        Calculate approximate codebook utilization.
        """

        if indices.numel() == 0:
            return torch.zeros(
                (),
                device=indices.device,
            )

        one_hot = F.one_hot(
            indices,
            num_classes=codebook_size,
        ).float()

        probabilities = one_hot.mean(dim=0)

        perplexity = torch.exp(
            -torch.sum(
                probabilities
                * torch.log(probabilities + 1e-10)
            )
        )

        return perplexity

    def forward(
        self,
        z: torch.Tensor,
        subject_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply H2D shared/private quantization.

        Parameters
        ----------
        z:
            Continuous latent tokens shaped:

                [batch_size, num_tokens, embedding_dim]

        subject_ids:
            Contiguous subject labels shaped:

                [batch_size]

            Valid IDs are between 0 and num_subjects - 1.
        """

        if z.ndim != 3:
            raise ValueError(
                "z must have shape "
                "[batch_size, num_tokens, embedding_dim]. "
                f"Received shape: {tuple(z.shape)}"
            )

        if subject_ids.ndim != 1:
            raise ValueError(
                "subject_ids must have shape [batch_size]."
            )

        if z.shape[0] != subject_ids.shape[0]:
            raise ValueError(
                "The batch size of z and subject_ids must match."
            )

        if z.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, but received "
                f"{z.shape[-1]}."
            )

        if subject_ids.numel() > 0:
            minimum_subject = int(
                subject_ids.min().item()
            )

            maximum_subject = int(
                subject_ids.max().item()
            )

            if minimum_subject < 0:
                raise ValueError(
                    "subject_ids cannot contain negative values."
                )

            if maximum_subject >= self.num_subjects:
                raise ValueError(
                    f"Received subject ID {maximum_subject}, but "
                    f"only {self.num_subjects} subjects exist."
                )

        batch_size, num_tokens, _ = z.shape

        # ------------------------------------------------------
        # Calculate shared-codebook assignments for every token
        # ------------------------------------------------------

        flat_tokens = z.reshape(
            -1,
            self.embedding_dim,
        )

        shared_distances_flat = self._compute_distances(
            flat_tokens,
            self.shared_codebook.weight,
        )

        shared_distances = shared_distances_flat.view(
            batch_size,
            num_tokens,
            self.shared_codebook_size,
        )

        minimum_shared_distances, shared_indices = (
            torch.min(
                shared_distances,
                dim=-1,
            )
        )

        # Select the closest shared_ratio fraction of tokens.
        shared_mask = self._select_shared_mask(
            minimum_shared_distances
        )

        private_mask = ~shared_mask

        # ------------------------------------------------------
        # Allocate outputs
        # ------------------------------------------------------

        quantized_raw = torch.zeros_like(z)

        private_indices = torch.full(
            (batch_size, num_tokens),
            fill_value=-1,
            dtype=torch.long,
            device=z.device,
        )

        # ------------------------------------------------------
        # Quantize shared tokens
        # ------------------------------------------------------

        shared_quantized_all = F.embedding(
            shared_indices,
            self.shared_codebook.weight,
        )

        quantized_raw[shared_mask] = (
            shared_quantized_all[shared_mask]
        )

        selected_shared_tokens = z.detach()[shared_mask]

        selected_shared_indices = shared_indices[
            shared_mask
        ]

        if self.training:
            self._update_shared_codebook(
                selected_tokens=selected_shared_tokens,
                selected_indices=selected_shared_indices,
            )

        # ------------------------------------------------------
        # Quantize private tokens using subject-specific codebooks
        # ------------------------------------------------------

        private_codebook_loss_sum = torch.zeros(
            (),
            device=z.device,
            dtype=z.dtype,
        )

        private_token_count = 0

        private_perplexities = []

        for subject_index in range(self.num_subjects):
            sample_mask = subject_ids == subject_index

            if not torch.any(sample_mask):
                continue

            subject_token_mask = (
                sample_mask.unsqueeze(1)
                & private_mask
            )

            subject_tokens = z[subject_token_mask]

            if subject_tokens.numel() == 0:
                continue

            private_codebook = self.private_codebooks[
                subject_index
            ]

            private_distances = self._compute_distances(
                subject_tokens,
                private_codebook.weight,
            )

            subject_private_indices = torch.argmin(
                private_distances,
                dim=1,
            )

            subject_private_quantized = private_codebook(
                subject_private_indices
            )

            quantized_raw[subject_token_mask] = (
                subject_private_quantized
            )

            private_indices[subject_token_mask] = (
                subject_private_indices
            )

            subject_private_loss = F.mse_loss(
                subject_private_quantized,
                subject_tokens.detach(),
                reduction="sum",
            )

            private_codebook_loss_sum = (
                private_codebook_loss_sum
                + subject_private_loss
            )

            private_token_count += subject_tokens.shape[0]

            subject_perplexity = (
                self._calculate_perplexity(
                    subject_private_indices,
                    self.private_codebook_size,
                )
            )

            private_perplexities.append(
                subject_perplexity
            )

        if private_token_count > 0:
            private_codebook_loss = (
                private_codebook_loss_sum
                / (
                    private_token_count
                    * self.embedding_dim
                )
            )
        else:
            private_codebook_loss = torch.zeros(
                (),
                device=z.device,
                dtype=z.dtype,
            )

        # ------------------------------------------------------
        # Commitment loss for all tokens
        # ------------------------------------------------------

        commitment_loss = F.mse_loss(
            z,
            quantized_raw.detach(),
        )

        quantization_loss = (
            private_codebook_loss
            + self.commitment_beta
            * commitment_loss
        )

        # Straight-through estimator.
        quantized = z + (
            quantized_raw - z
        ).detach()

        shared_perplexity = (
            self._calculate_perplexity(
                selected_shared_indices,
                self.shared_codebook_size,
            )
        )

        if private_perplexities:
            private_perplexity = torch.stack(
                private_perplexities
            ).mean()
        else:
            private_perplexity = torch.zeros(
                (),
                device=z.device,
            )

        shared_tokens_per_sample = shared_mask.sum(
            dim=1
        )

        private_tokens_per_sample = private_mask.sum(
            dim=1
        )

        return {
            "quantized": quantized,
            "quantized_raw": quantized_raw,
            "shared_mask": shared_mask,
            "private_mask": private_mask,
            "shared_indices": shared_indices,
            "private_indices": private_indices,
            "minimum_shared_distances": (
                minimum_shared_distances
            ),
            "private_codebook_loss": (
                private_codebook_loss
            ),
            "commitment_loss": commitment_loss,
            "loss": quantization_loss,
            "shared_perplexity": shared_perplexity,
            "private_perplexity": private_perplexity,
            "shared_tokens_per_sample": (
                shared_tokens_per_sample
            ),
            "private_tokens_per_sample": (
                private_tokens_per_sample
            ),
        }
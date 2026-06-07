"""Self-attention feature extractor for combat observations."""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sts2_env.training.entity_tokens import (
    EntityTokenProjections,
    NUM_NODES,
    masked_mean_pool,
)


class CombatAttentionExtractor(BaseFeaturesExtractor):
    """Parse flat obs v3 into entity tokens, run self-attention, pool."""

    def __init__(
        self,
        observation_space: spaces.Space,
        *,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        self.d_model = d_model
        self.n_tokens = NUM_NODES
        self.projections = EntityTokenProjections(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        tokens, valid_mask = self.projections.build_entity_tokens(observations)
        key_padding_mask = ~valid_mask
        encoded = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        pooled = masked_mean_pool(encoded, valid_mask)
        return self.output_proj(pooled)

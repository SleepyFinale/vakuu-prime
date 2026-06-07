"""Graph neural network feature extractor for combat observations."""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sts2_env.training.entity_graph import build_adjacency
from sts2_env.training.entity_tokens import (
    EntityTokenProjections,
    NUM_NODES,
    masked_mean_pool,
)


class CombatGNNExtractor(BaseFeaturesExtractor):
    """Parse flat obs v3 into entity tokens, run dense GAT layers, pool."""

    def __init__(
        self,
        observation_space: spaces.Space,
        *,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        self.d_model = d_model
        self.n_tokens = NUM_NODES
        self.projections = EntityTokenProjections(d_model)
        self.dropout = nn.Dropout(dropout)

        try:
            from torch_geometric.nn.dense import DenseGATConv
        except ImportError as exc:
            raise ImportError(
                "CombatGNNExtractor requires torch-geometric. "
                "Install with: pip install 'sts2-rl-agent[train]'"
            ) from exc

        self.gnn_layers = nn.ModuleList([
            DenseGATConv(d_model, d_model, heads=n_heads, concat=False)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        tokens, valid_mask = self.projections.build_entity_tokens(observations)
        adj = build_adjacency(observations, valid_mask)
        h = tokens
        for layer in self.gnn_layers:
            h = layer(h, adj)
            h = torch.relu(h)
            h = self.dropout(h)
        pooled = masked_mean_pool(h, valid_mask)
        return self.output_proj(pooled)

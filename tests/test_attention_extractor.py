"""Tests for CombatAttentionExtractor."""

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from gymnasium import spaces

from sts2_env.gym_env.observation import OBS_SIZE
from sts2_env.training.attention_extractor import CombatAttentionExtractor


def _make_extractor(features_dim: int = 256) -> CombatAttentionExtractor:
    obs_space = spaces.Box(low=-1.0, high=10.0, shape=(OBS_SIZE,), dtype=np.float32)
    return CombatAttentionExtractor(
        obs_space,
        d_model=64,
        n_heads=4,
        n_layers=1,
        features_dim=features_dim,
    )


def test_forward_output_shape():
    extractor = _make_extractor(features_dim=128)
    batch = torch.randn(4, OBS_SIZE)
    out = extractor(batch)
    assert out.shape == (4, 128)


def test_padding_mask_allows_sparse_entities():
    extractor = _make_extractor()
    obs = torch.zeros(2, OBS_SIZE)
    obs[:, 0] = 0.8
    obs[:, 2] = 0.5
    obs[:, 3] = 0.3
    out = extractor(obs)
    assert out.shape == (2, 256)
    assert torch.isfinite(out).all()


def test_gradient_flows():
    extractor = _make_extractor(features_dim=64)
    obs = torch.randn(2, OBS_SIZE, requires_grad=False)
    out = extractor(obs)
    loss = out.sum()
    loss.backward()
    assert extractor.projections.player_proj.weight.grad is not None

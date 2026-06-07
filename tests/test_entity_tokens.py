"""Tests for shared entity token builder."""

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from gymnasium import spaces

from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.characters.all import get_character
from sts2_env.core.combat import CombatState
from sts2_env.gym_env.observation import OBS_SIZE, encode_observation
from sts2_env.training.attention_extractor import CombatAttentionExtractor
from sts2_env.training.entity_tokens import EntityTokenProjections, NUM_NODES


def _make_combat() -> CombatState:
    char_cfg = get_character("Ironclad")
    combat = CombatState(
        player_hp=char_cfg.starting_hp,
        player_max_hp=char_cfg.starting_hp,
        deck=create_ironclad_starter_deck(),
        rng_seed=42,
        character_id="Ironclad",
        relics=[char_cfg.starting_relic],
    )
    combat.start_combat()
    return combat


def test_entity_tokens_shape_matches_num_nodes():
    projections = EntityTokenProjections(d_model=32)
    obs = torch.from_numpy(encode_observation(_make_combat())).unsqueeze(0)
    tokens, mask = projections.build_entity_tokens(obs)
    assert tokens.shape == (1, NUM_NODES, 32)
    assert mask.shape == (1, NUM_NODES)
    assert mask[:, 0].all()
    assert mask[:, 1].all()
    assert mask[:, 2].all()


def test_shared_tokens_match_attention_masks():
    obs_np = encode_observation(_make_combat())
    obs = torch.from_numpy(obs_np).unsqueeze(0)

    projections = EntityTokenProjections(d_model=64)
    _, shared_mask = projections.build_entity_tokens(obs)

    extractor = CombatAttentionExtractor(
        spaces.Box(low=-1.0, high=10.0, shape=(OBS_SIZE,), dtype=np.float32),
        d_model=64,
        n_heads=4,
        n_layers=1,
        features_dim=64,
    )
    with torch.no_grad():
        _, attn_mask = extractor.projections.build_entity_tokens(obs)

    assert torch.equal(shared_mask, attn_mask)

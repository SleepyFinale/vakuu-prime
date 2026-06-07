"""Tests for structural combat graph adjacency."""

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.characters.all import get_character
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import TargetType
from sts2_env.gym_env.observation import CARD_IDS, NUM_CARD_IDS, encode_observation
from sts2_env.training.card_target_table import card_target_by_index
from sts2_env.training.entity_graph import build_adjacency
from sts2_env.training.entity_tokens import (
    EntityTokenProjections,
    NODE_CARDS_START,
    NODE_ENEMIES_START,
    NODE_MECHANICS,
    NODE_PILES,
    NODE_PLAYER,
    NODE_RELICS_START,
    NUM_NODES,
)


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


def test_adjacency_shape():
    obs = torch.from_numpy(encode_observation(_make_combat())).unsqueeze(0)
    projections = EntityTokenProjections(d_model=16)
    _, mask = projections.build_entity_tokens(obs)
    adj = build_adjacency(obs, mask)
    assert adj.shape == (1, NUM_NODES, NUM_NODES)


def test_pile_and_relic_edges_to_player():
    obs = torch.from_numpy(encode_observation(_make_combat())).unsqueeze(0)
    projections = EntityTokenProjections(d_model=16)
    _, mask = projections.build_entity_tokens(obs)
    adj = build_adjacency(obs, mask)[0]

    assert adj[NODE_PILES, NODE_PLAYER] > 0
    assert adj[NODE_PLAYER, NODE_PILES] > 0
    assert adj[NODE_RELICS_START, NODE_PLAYER] > 0


def test_strike_card_edges_to_alive_enemy():
    targets = card_target_by_index()
    strike_index = next(i for i, card_id in enumerate(CARD_IDS) if card_id.name == "STRIKE_IRONCLAD")
    assert targets[strike_index] in (TargetType.ANY_ENEMY, TargetType.RANDOM_ENEMY, TargetType.ALL_ENEMIES)

    obs = torch.zeros(1, encode_observation(_make_combat()).shape[0])
    mask = torch.zeros(1, NUM_NODES, dtype=torch.bool)

    card_node = NODE_CARDS_START
    enemy_node = NODE_ENEMIES_START
    hand_base = 10
    obs[0, hand_base + 0] = (strike_index + 1) / (NUM_CARD_IDS + 1)
    obs[0, hand_base + 1] = 1.0
    mask[0, NODE_PLAYER] = True
    mask[0, NODE_PILES] = True
    mask[0, NODE_MECHANICS] = True
    mask[0, card_node] = True
    mask[0, enemy_node] = True

    enemy_base = 66
    obs[0, enemy_base + 0] = 1.0

    adj = build_adjacency(obs, mask)[0]
    assert adj[card_node, enemy_node] > 0


def test_enemy_attack_intent_edges_to_player():
    obs = torch.zeros(1, encode_observation(_make_combat()).shape[0])
    mask = torch.zeros(1, NUM_NODES, dtype=torch.bool)
    mask[0, NODE_PLAYER] = True
    mask[0, NODE_PILES] = True
    mask[0, NODE_MECHANICS] = True
    enemy_node = NODE_ENEMIES_START
    mask[0, enemy_node] = True

    enemy_base = 66
    obs[0, enemy_base + 0] = 1.0
    obs[0, enemy_base + 3] = 1.0

    adj = build_adjacency(obs, mask)[0]
    assert adj[enemy_node, NODE_PLAYER] > 0


def test_invalid_nodes_have_no_external_edges():
    obs = torch.zeros(1, encode_observation(_make_combat()).shape[0])
    mask = torch.zeros(1, NUM_NODES, dtype=torch.bool)
    mask[0, NODE_PLAYER] = True
    mask[0, NODE_PILES] = True
    mask[0, NODE_MECHANICS] = True

    adj = build_adjacency(obs, mask)[0]
    card_node = NODE_CARDS_START
    assert adj[card_node, :].sum() == 0
    assert adj[:, card_node].sum() == 0

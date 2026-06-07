"""Structural combat graph adjacency for the GNN feature extractor."""

from __future__ import annotations

from functools import lru_cache

import torch

from sts2_env.core.constants import MAX_ENEMIES, MAX_HAND_SIZE
from sts2_env.core.enums import TargetType
from sts2_env.gym_env.observation import (
    CARD_FEATURES,
    ENEMY_FEATURES,
    MAX_RELIC_SLOTS,
    NUM_CARD_IDS,
    TOKEN_SLICES,
)
from sts2_env.training.card_target_table import card_target_by_index
from sts2_env.training.entity_tokens import (
    NODE_CARDS_START,
    NODE_ENEMIES_START,
    NODE_PILES,
    NODE_PLAYER,
    NODE_RELICS_START,
    NUM_NODES,
)

_ENEMY_INTENT_START = 3
_ATTACK_INTENT_INDEX = 0
_MULTI_ATTACK_INTENT_INDEX = 1


@lru_cache(maxsize=1)
def _target_type_value_table() -> tuple[int, ...]:
    return tuple(target.value for target in card_target_by_index())


def _norms_to_indices(norms: torch.Tensor) -> torch.Tensor:
    """Map card_id_norm batch to CARD_IDS indices (-1 when empty/invalid)."""
    scaled = norms * float(NUM_CARD_IDS + 1)
    indices = torch.round(scaled).to(torch.long) - 1
    invalid = (norms <= 0) | (indices < 0) | (indices >= NUM_CARD_IDS)
    return indices.masked_fill(invalid, -1)


def build_adjacency(obs: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Build batched dense adjacency (B, N, N) from obs v3 and node validity."""
    batch_size = obs.shape[0]
    device = obs.device
    adj = torch.zeros(batch_size, NUM_NODES, NUM_NODES, device=device, dtype=obs.dtype)

    adj[:, NODE_PILES, NODE_PLAYER] = 1.0

    hand_start, hand_end = TOKEN_SLICES["hand"]
    hand = obs[:, hand_start:hand_end].reshape(batch_size, MAX_HAND_SIZE, CARD_FEATURES)

    enemy_start, enemy_end = TOKEN_SLICES["enemies"]
    enemies = obs[:, enemy_start:enemy_end].reshape(batch_size, MAX_ENEMIES, ENEMY_FEATURES)

    target_table = torch.tensor(
        _target_type_value_table(),
        device=device,
        dtype=torch.long,
    )

    self_val = TargetType.SELF.value
    none_val = TargetType.NONE.value
    any_enemy_val = TargetType.ANY_ENEMY.value
    random_enemy_val = TargetType.RANDOM_ENEMY.value
    all_enemies_val = TargetType.ALL_ENEMIES.value

    for card_slot in range(MAX_HAND_SIZE):
        card_node = NODE_CARDS_START + card_slot
        norms = hand[:, card_slot, 0]
        indices = _norms_to_indices(norms)
        card_valid = valid_mask[:, card_node] & (indices >= 0)
        if not card_valid.any():
            continue

        target_vals = target_table[indices.clamp(min=0)]

        to_player = card_valid & (
            (target_vals == self_val) | (target_vals == none_val)
        )
        adj[to_player, card_node, NODE_PLAYER] = 1.0

        for enemy_slot in range(MAX_ENEMIES):
            enemy_node = NODE_ENEMIES_START + enemy_slot
            enemy_valid = valid_mask[:, enemy_node]
            to_enemy = card_valid & enemy_valid & (
                (target_vals == all_enemies_val)
                | (target_vals == any_enemy_val)
                | (target_vals == random_enemy_val)
            )
            adj[to_enemy, card_node, enemy_node] = 1.0

    for enemy_slot in range(MAX_ENEMIES):
        enemy_node = NODE_ENEMIES_START + enemy_slot
        enemy = enemies[:, enemy_slot, :]
        enemy_valid = valid_mask[:, enemy_node]
        if not enemy_valid.any():
            continue

        attack_intent = enemy[:, _ENEMY_INTENT_START + _ATTACK_INTENT_INDEX] > 0
        multi_attack_intent = enemy[:, _ENEMY_INTENT_START + _MULTI_ATTACK_INTENT_INDEX] > 0
        threatens = enemy_valid & (attack_intent | multi_attack_intent)
        adj[threatens, enemy_node, NODE_PLAYER] = 1.0
        adj[enemy_valid, NODE_PLAYER, enemy_node] = 1.0

    for relic_slot in range(MAX_RELIC_SLOTS):
        relic_node = NODE_RELICS_START + relic_slot
        relic_valid = valid_mask[:, relic_node]
        if relic_valid.any():
            adj[relic_valid, relic_node, NODE_PLAYER] = 1.0

    adj = adj + adj.transpose(1, 2)
    adj = (adj > 0).to(obs.dtype)

    node_mask = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)
    adj = adj * node_mask.to(adj.dtype)
    return adj

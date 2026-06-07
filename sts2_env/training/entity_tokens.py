"""Shared entity tokenization for combat feature extractors."""

from __future__ import annotations

import torch
import torch.nn as nn

from sts2_env.core.constants import MAX_ENEMIES, MAX_HAND_SIZE
from sts2_env.gym_env.observation import (
    CARD_FEATURES,
    CHARACTER_MECHANICS_FEATURES,
    ENEMY_FEATURES,
    MAX_POTION_OBS_SLOTS,
    MAX_RELIC_SLOTS,
    NUM_PLAYER_POWERS,
    PILE_FEATURES,
    PLAYER_CORE_FEATURES,
    POTION_FEATURES,
    RELIC_FEATURES,
    TOKEN_SLICES,
)

NUM_ENTITY_TYPES = 7
ENTITY_TYPE_PLAYER = 0
ENTITY_TYPE_PILE = 1
ENTITY_TYPE_MECHANICS = 2
ENTITY_TYPE_CARD = 3
ENTITY_TYPE_ENEMY = 4
ENTITY_TYPE_RELIC = 5
ENTITY_TYPE_POTION = 6

PLAYER_TOKEN_FEATURES = PLAYER_CORE_FEATURES + NUM_PLAYER_POWERS
NUM_NODES = (
    1 + 1 + 1 + MAX_HAND_SIZE + MAX_ENEMIES + MAX_RELIC_SLOTS + MAX_POTION_OBS_SLOTS
)

NODE_PLAYER = 0
NODE_PILES = 1
NODE_MECHANICS = 2
NODE_CARDS_START = 3
NODE_ENEMIES_START = NODE_CARDS_START + MAX_HAND_SIZE
NODE_RELICS_START = NODE_ENEMIES_START + MAX_ENEMIES
NODE_POTIONS_START = NODE_RELICS_START + MAX_RELIC_SLOTS

NODE_OFFSETS = {
    "player": NODE_PLAYER,
    "piles": NODE_PILES,
    "mechanics": NODE_MECHANICS,
    "cards": NODE_CARDS_START,
    "enemies": NODE_ENEMIES_START,
    "relics": NODE_RELICS_START,
    "potions": NODE_POTIONS_START,
}

_TYPE_IDS = (
    [ENTITY_TYPE_PLAYER]
    + [ENTITY_TYPE_PILE]
    + [ENTITY_TYPE_MECHANICS]
    + [ENTITY_TYPE_CARD] * MAX_HAND_SIZE
    + [ENTITY_TYPE_ENEMY] * MAX_ENEMIES
    + [ENTITY_TYPE_RELIC] * MAX_RELIC_SLOTS
    + [ENTITY_TYPE_POTION] * MAX_POTION_OBS_SLOTS
)


class EntityTokenProjections(nn.Module):
    """Per-entity-type linear projections and type embeddings."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.player_proj = nn.Linear(PLAYER_TOKEN_FEATURES, d_model)
        self.pile_proj = nn.Linear(PILE_FEATURES, d_model)
        self.mechanics_proj = nn.Linear(CHARACTER_MECHANICS_FEATURES, d_model)
        self.card_proj = nn.Linear(CARD_FEATURES, d_model)
        self.enemy_proj = nn.Linear(ENEMY_FEATURES, d_model)
        self.relic_proj = nn.Linear(RELIC_FEATURES, d_model)
        self.potion_proj = nn.Linear(POTION_FEATURES, d_model)
        self.type_embeddings = nn.Embedding(NUM_ENTITY_TYPES, d_model)
        self.register_buffer(
            "_type_ids",
            torch.tensor(_TYPE_IDS, dtype=torch.long),
            persistent=False,
        )

    def build_entity_tokens(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return token embeddings (B, N, d_model) and validity mask (B, N)."""
        batch_size = obs.shape[0]
        tokens: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        player_start, player_end = TOKEN_SLICES["player"]
        player = obs[:, player_start:player_end]
        tokens.append(self.player_proj(player).unsqueeze(1))
        masks.append(torch.ones(batch_size, 1, device=obs.device, dtype=torch.bool))

        pile_start, pile_end = TOKEN_SLICES["piles"]
        piles = obs[:, pile_start:pile_end]
        tokens.append(self.pile_proj(piles).unsqueeze(1))
        masks.append(torch.ones(batch_size, 1, device=obs.device, dtype=torch.bool))

        mech_start, mech_end = TOKEN_SLICES["mechanics"]
        mechanics = obs[:, mech_start:mech_end]
        tokens.append(self.mechanics_proj(mechanics).unsqueeze(1))
        masks.append(torch.ones(batch_size, 1, device=obs.device, dtype=torch.bool))

        hand_start, hand_end = TOKEN_SLICES["hand"]
        hand = obs[:, hand_start:hand_end].reshape(batch_size, MAX_HAND_SIZE, CARD_FEATURES)
        card_tokens = self.card_proj(hand)
        card_mask = hand[:, :, 0] > 0
        tokens.append(card_tokens)
        masks.append(card_mask)

        enemy_start, enemy_end = TOKEN_SLICES["enemies"]
        enemies = obs[:, enemy_start:enemy_end].reshape(batch_size, MAX_ENEMIES, ENEMY_FEATURES)
        enemy_tokens = self.enemy_proj(enemies)
        enemy_mask = enemies[:, :, 0] > 0
        tokens.append(enemy_tokens)
        masks.append(enemy_mask)

        relic_start, relic_end = TOKEN_SLICES["relics"]
        relics = obs[:, relic_start:relic_end].reshape(batch_size, MAX_RELIC_SLOTS, RELIC_FEATURES)
        relic_tokens = self.relic_proj(relics)
        relic_mask = relics.abs().sum(dim=-1) > 0
        tokens.append(relic_tokens)
        masks.append(relic_mask)

        potion_start, potion_end = TOKEN_SLICES["potions"]
        potions = obs[:, potion_start:potion_end].reshape(
            batch_size, MAX_POTION_OBS_SLOTS, POTION_FEATURES
        )
        potion_tokens = self.potion_proj(potions)
        potion_mask = potions.abs().sum(dim=-1) > 0
        tokens.append(potion_tokens)
        masks.append(potion_mask)

        token_tensor = torch.cat(tokens, dim=1)
        mask_tensor = torch.cat(masks, dim=1)
        type_ids = self._type_ids.to(obs.device)
        token_tensor = token_tensor + self.type_embeddings(type_ids).unsqueeze(0)
        return token_tensor, mask_tensor


def masked_mean_pool(tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings over valid nodes."""
    mask_f = valid_mask.unsqueeze(-1).to(tokens.dtype)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (tokens * mask_f).sum(dim=1) / denom

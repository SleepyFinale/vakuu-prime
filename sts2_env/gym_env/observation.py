"""Observation space encoding.

Compact flat float32 vector (148 dimensions):
  Player state:       hp/max_hp, block/50, energy, max_energy       (4)
  Player powers:      str, dex, vuln, weak, frail, artifact         (6)
  Hand (10 cards):    card_id_norm, cost, damage, block, is_attack  (50)
  Pile sizes:         draw, discard, exhaust, reserved, reserved,
                      reserved                                     (6)
  Enemies (5 slots):  alive, hp%, block, intent_onehot(5),
                      intent_dmg, intent_hits, vuln, weak, str      (13 * 5 = 65)
  Character mechanics: one-hot(5), stars, orb cap/count, orbs(3*2), osty(3) (17)
Total: 4 + 6 + 50 + 6 + 65 + 17 = 148
"""

from __future__ import annotations

import numpy as np

from sts2_env.characters.all import SUPPORTED_TRAINING_CHARACTERS
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import CardId, IntentType, OrbType, PowerId
from sts2_env.core.constants import MAX_HAND_SIZE, MAX_ENEMIES
from sts2_env.orbs.base import OrbQueue

# Card IDs list for normalised encoding
CARD_IDS = list(CardId)
NUM_CARD_IDS = len(CARD_IDS)
_CARD_ID_TO_IDX = {cid: i for i, cid in enumerate(CARD_IDS)}

# Player powers to track (6)
PLAYER_POWERS = [
    PowerId.STRENGTH, PowerId.DEXTERITY, PowerId.VULNERABLE,
    PowerId.WEAK, PowerId.FRAIL, PowerId.ARTIFACT,
]
NUM_PLAYER_POWERS = len(PLAYER_POWERS)

# Intent types for one-hot (5)
INTENT_TYPES = [
    IntentType.ATTACK, IntentType.MULTI_ATTACK, IntentType.DEFEND,
    IntentType.BUFF, IntentType.DEBUFF,
]
NUM_INTENT_TYPES = len(INTENT_TYPES)

# Per-card features in hand
CARD_FEATURES = 5  # card_id_norm, cost, damage, block, is_attack

# Per-enemy features
# alive(1) + hp%(1) + block(1) + intent_onehot(5) + intent_dmg(1) + intent_hits(1) + vuln(1) + weak(1) + str(1)
ENEMY_FEATURES = 1 + 1 + 1 + NUM_INTENT_TYPES + 1 + 1 + 1 + 1 + 1  # = 13

# Pile summary features
PILE_FEATURES = 6  # draw_size, discard_size, exhaust_size, reserved x3

# Character mechanics (orbs, stars, Osty companion)
NUM_TRAINING_CHARACTERS = len(SUPPORTED_TRAINING_CHARACTERS)
CHARACTER_ONE_HOT_FEATURES = NUM_TRAINING_CHARACTERS
STARS_FEATURES = 1
ORB_SUMMARY_FEATURES = 2  # capacity, count
ORB_SLOT_FEATURES = 3 * 2  # type_index, evoke_value per slot (first 3 slots)
OSTY_FEATURES = 3  # alive, hp_ratio, block
CHARACTER_MECHANICS_FEATURES = (
    CHARACTER_ONE_HOT_FEATURES
    + STARS_FEATURES
    + ORB_SUMMARY_FEATURES
    + ORB_SLOT_FEATURES
    + OSTY_FEATURES
)  # = 17

_CHARACTER_ID_TO_ONE_HOT: dict[str, int] = {
    char_id: index for index, char_id in enumerate(SUPPORTED_TRAINING_CHARACTERS)
}

_ORB_TYPE_NAME_TO_INDEX: dict[str, int] = {
    orb_type.name: index for index, orb_type in enumerate(OrbType)
}

# Base combat observation (before character mechanics)
BASE_OBS_SIZE = (
    4                                  # player state
    + NUM_PLAYER_POWERS                # player powers (6)
    + MAX_HAND_SIZE * CARD_FEATURES    # hand cards (50)
    + PILE_FEATURES                    # pile summaries (6)
    + MAX_ENEMIES * ENEMY_FEATURES     # enemies (65)
)  # = 131

# Full observation size
OBS_SIZE = BASE_OBS_SIZE + CHARACTER_MECHANICS_FEATURES  # = 148


def encode_character_mechanics_from_fields(
    obs: np.ndarray,
    start_idx: int,
    *,
    character_id: str | None = None,
    stars: int = 0,
    orb_capacity: int = 0,
    orb_count: int = 0,
    orbs: list[tuple[str, int]] | None = None,
    osty_alive: bool = False,
    osty_hp: int = 0,
    osty_max_hp: int = 0,
    osty_block: int = 0,
) -> int:
    """Encode character mechanics from plain fields (simulator or bridge JSON)."""
    idx = start_idx

    if character_id is not None:
        char_index = _CHARACTER_ID_TO_ONE_HOT.get(character_id)
        if char_index is None:
            normalized = character_id.strip()
            for known_id, known_index in _CHARACTER_ID_TO_ONE_HOT.items():
                if known_id.lower() == normalized.lower():
                    char_index = known_index
                    break
        if char_index is not None:
            obs[idx + char_index] = 1.0
    idx += CHARACTER_ONE_HOT_FEATURES

    obs[idx] = stars / 30.0
    idx += STARS_FEATURES

    obs[idx] = orb_capacity / OrbQueue.MAX_CAPACITY
    obs[idx + 1] = orb_count / OrbQueue.MAX_CAPACITY
    orb_entries = orbs or []
    for slot in range(3):
        slot_idx = idx + ORB_SUMMARY_FEATURES + slot * 2
        if slot < len(orb_entries):
            orb_type_name, evoke_value = orb_entries[slot]
            type_index = _ORB_TYPE_NAME_TO_INDEX.get(orb_type_name.upper(), 0)
            obs[slot_idx] = (type_index + 1) / len(OrbType)
            obs[slot_idx + 1] = evoke_value / 50.0
    idx += ORB_SUMMARY_FEATURES + ORB_SLOT_FEATURES

    if osty_alive and osty_max_hp > 0:
        obs[idx] = 1.0
        obs[idx + 1] = osty_hp / osty_max_hp
        obs[idx + 2] = osty_block / 50.0
    idx += OSTY_FEATURES

    return idx


def _encode_character_mechanics(combat: CombatState, obs: np.ndarray, start_idx: int) -> int:
    """Encode character-specific mechanics into obs starting at start_idx."""
    orb_entries: list[tuple[str, int]] = []
    orb_queue = combat.orb_queue
    orb_capacity = 0
    orb_count = 0
    if isinstance(orb_queue, OrbQueue):
        orb_capacity = orb_queue.capacity
        orb_count = len(orb_queue.orbs)
        for orb in orb_queue.orbs[:3]:
            orb_entries.append((
                orb.orb_type.name,
                orb.get_evoke_value(combat),
            ))

    osty = combat.get_osty()
    return encode_character_mechanics_from_fields(
        obs,
        start_idx,
        character_id=combat.character_id,
        stars=combat.stars,
        orb_capacity=orb_capacity,
        orb_count=orb_count,
        orbs=orb_entries,
        osty_alive=osty is not None and osty.is_alive,
        osty_hp=osty.current_hp if osty is not None else 0,
        osty_max_hp=osty.max_hp if osty is not None else 0,
        osty_block=osty.block if osty is not None else 0,
    )


def encode_observation(combat: CombatState) -> np.ndarray:
    """Encode combat state as a compact flat float32 vector."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    idx = 0

    # --- Player state (4) ---
    obs[idx] = combat.player.current_hp / combat.player.max_hp if combat.player.max_hp > 0 else 0.0
    obs[idx + 1] = combat.player.block / 50.0
    obs[idx + 2] = combat.energy / 10.0
    obs[idx + 3] = combat.max_energy / 10.0
    idx += 4

    # --- Player powers (6) ---
    for pid in PLAYER_POWERS:
        obs[idx] = combat.player.get_power_amount(pid) / 20.0
        idx += 1

    # --- Hand cards (10 * 5 = 50) ---
    for i in range(MAX_HAND_SIZE):
        if i < len(combat.hand):
            card = combat.hand[i]
            obs[idx] = (_CARD_ID_TO_IDX.get(card.card_id, 0) + 1) / (NUM_CARD_IDS + 1)
            obs[idx + 1] = max(0, card.cost) / 5.0
            obs[idx + 2] = (card.base_damage or 0) / 50.0
            obs[idx + 3] = (card.base_block or 0) / 50.0
            obs[idx + 4] = 1.0 if card.is_attack else 0.0
        idx += CARD_FEATURES

    # --- Pile summaries (6) ---
    obs[idx] = len(combat.draw_pile) / 20.0
    obs[idx + 1] = len(combat.discard_pile) / 20.0
    obs[idx + 2] = len(combat.exhaust_pile) / 20.0
    # Keep the last three pile-summary dimensions zeroed so simulator and
    # bridge observations stay aligned even though the bridge only exposes
    # aggregate pile counts.
    obs[idx + 3] = 0.0
    obs[idx + 4] = 0.0
    obs[idx + 5] = 0.0
    idx += PILE_FEATURES

    # --- Enemies (5 * 13 = 65) ---
    for i in range(MAX_ENEMIES):
        if i < len(combat.enemies):
            enemy = combat.enemies[i]
            obs[idx] = 1.0 if enemy.is_alive else 0.0
            obs[idx + 1] = enemy.current_hp / enemy.max_hp if enemy.max_hp > 0 else 0.0
            obs[idx + 2] = enemy.block / 50.0

            # Intent encoding (one-hot + damage + hits)
            ai = combat.enemy_ais.get(enemy.combat_id)
            if ai is not None and enemy.is_alive:
                move = ai.current_move
                if move.intents:
                    intent = move.intents[0]
                    for j, it in enumerate(INTENT_TYPES):
                        if intent.intent_type == it:
                            obs[idx + 3 + j] = 1.0
                    obs[idx + 3 + NUM_INTENT_TYPES] = intent.damage / 30.0
                    obs[idx + 3 + NUM_INTENT_TYPES + 1] = intent.hits / 5.0

            # Enemy powers
            obs[idx + 3 + NUM_INTENT_TYPES + 2] = enemy.get_power_amount(PowerId.VULNERABLE) / 10.0
            obs[idx + 3 + NUM_INTENT_TYPES + 3] = enemy.get_power_amount(PowerId.WEAK) / 10.0
            obs[idx + 3 + NUM_INTENT_TYPES + 4] = enemy.get_power_amount(PowerId.STRENGTH) / 10.0
        idx += ENEMY_FEATURES

    _encode_character_mechanics(combat, obs, idx)

    return obs

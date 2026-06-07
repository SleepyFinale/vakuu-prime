"""Tests for full combat power observation encoding (obs v6)."""

from __future__ import annotations

import pytest

from sts2_env.bridge.state_adapter import StateAdapter
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.characters.all import get_character
from sts2_env.core.combat import CombatState
from sts2_env.core.constants import MAX_ENEMIES
from sts2_env.core.enums import PowerId
from sts2_env.core.rng import Rng
from sts2_env.gym_env.observation import (
    COMBAT_OBS_V2_SIZE,
    ENEMY_CORE_FEATURES,
    ENEMY_FEATURES,
    ENEMY_POWERS,
    NUM_ENEMY_POWERS,
    NUM_PLAYER_POWERS,
    OBS_SIZE,
    PLAYER_CORE_FEATURES,
    PLAYER_POWERS,
    POTION_OBS_SIZE,
    RELIC_OBS_SIZE,
    TOKEN_SLICES,
    _POWER_ID_TO_ENEMY_IDX,
    _POWER_ID_TO_PLAYER_IDX,
    encode_observation,
)
from sts2_env.monsters.act1_weak import create_fuzzy_wurm_crawler


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
    rng = Rng(42)
    creature, ai = create_fuzzy_wurm_crawler(rng)
    combat.add_enemy(creature, ai)
    combat.start_combat()
    return combat


def test_obs_v6_size_constants() -> None:
    assert NUM_PLAYER_POWERS == 268
    assert NUM_ENEMY_POWERS == 268
    assert ENEMY_FEATURES == ENEMY_CORE_FEATURES + NUM_ENEMY_POWERS
    assert OBS_SIZE == COMBAT_OBS_V2_SIZE + RELIC_OBS_SIZE + POTION_OBS_SIZE
    assert OBS_SIZE == 1985


def test_player_powers_encode_at_correct_indices() -> None:
    combat = _make_combat()
    player_cases = {
        PowerId.BARRICADE: 1,
        PowerId.CORRUPTION: 1,
        PowerId.ECHO_FORM: 2,
        PowerId.POISON: 12,
        PowerId.DOOM: 7,
    }
    for pid, amount in player_cases.items():
        combat.player.apply_power(pid, amount)

    obs = encode_observation(combat)
    for pid, amount in player_cases.items():
        idx = PLAYER_CORE_FEATURES + _POWER_ID_TO_PLAYER_IDX[pid]
        assert obs[idx] == pytest.approx(amount / 20.0)


def test_enemy_powers_encode_at_correct_indices() -> None:
    combat = _make_combat()
    enemy = combat.enemies[0]
    enemy_cases = {
        PowerId.RITUAL: 3,
        PowerId.ARTIFACT: 2,
        PowerId.REGEN: 5,
        PowerId.THORNS: 4,
    }
    for pid, amount in enemy_cases.items():
        enemy.apply_power(pid, amount)

    obs = encode_observation(combat)
    enemy_start, _ = TOKEN_SLICES["enemies"]
    slot_start = enemy_start
    power_base = slot_start + ENEMY_CORE_FEATURES
    for pid, amount in enemy_cases.items():
        idx = power_base + _POWER_ID_TO_ENEMY_IDX[pid]
        assert obs[idx] == pytest.approx(amount / 10.0)


def test_untracked_powers_encode_as_zero() -> None:
    combat = _make_combat()
    obs = encode_observation(combat)
    player_power_start = PLAYER_CORE_FEATURES
    player_power_end = player_power_start + NUM_PLAYER_POWERS
    assert obs[player_power_start:player_power_end].sum() == pytest.approx(0.0)

    enemy_start, enemy_end = TOKEN_SLICES["enemies"]
    for slot in range(MAX_ENEMIES):
        power_slice = obs[
            enemy_start + slot * ENEMY_FEATURES + ENEMY_CORE_FEATURES:
            enemy_start + (slot + 1) * ENEMY_FEATURES
        ]
        assert power_slice.sum() == pytest.approx(0.0)
    assert enemy_end - enemy_start == MAX_ENEMIES * ENEMY_FEATURES


def test_bridge_adapter_encodes_player_powers() -> None:
    adapter = StateAdapter()
    state = {
        "type": "combat_action",
        "player": {
            "hp": 70,
            "max_hp": 80,
            "block": 0,
            "energy": 3,
            "max_energy": 3,
            "powers": [
                {"id": "BARRICADE", "amount": 1},
                {"id": "METALLICIZE", "amount": 3},
            ],
            "character_id": "Ironclad",
            "stars": 0,
            "orb_queue": {"capacity": 0, "count": 0, "orbs": []},
            "osty": {"alive": False, "hp": 0, "max_hp": 0, "block": 0},
        },
        "hand": [],
        "enemies": [
            {
                "id": "NIBBIT",
                "hp": 30,
                "max_hp": 30,
                "block": 0,
                "is_alive": True,
                "intent": "ATTACK",
                "intent_damage": 6,
                "intent_hits": 1,
                "powers": [{"id": "RITUAL", "amount": 2}],
            }
        ],
        "draw_pile_count": 0,
        "discard_pile_count": 0,
        "exhaust_pile_count": 0,
    }
    obs = adapter.encode_observation(state)
    assert obs.shape == (OBS_SIZE,)

    barricade_idx = PLAYER_CORE_FEATURES + _POWER_ID_TO_PLAYER_IDX[PowerId.BARRICADE]
    metallicize_idx = PLAYER_CORE_FEATURES + _POWER_ID_TO_PLAYER_IDX[PowerId.METALLICIZE]
    assert obs[barricade_idx] == pytest.approx(1 / 20.0)
    assert obs[metallicize_idx] == pytest.approx(3 / 20.0)

    enemy_start, _ = TOKEN_SLICES["enemies"]
    ritual_idx = enemy_start + ENEMY_CORE_FEATURES + _POWER_ID_TO_ENEMY_IDX[PowerId.RITUAL]
    assert obs[ritual_idx] == pytest.approx(2 / 10.0)


def test_player_and_enemy_power_lists_match() -> None:
    assert PLAYER_POWERS == ENEMY_POWERS

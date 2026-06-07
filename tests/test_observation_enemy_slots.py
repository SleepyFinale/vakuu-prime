"""Regression tests for fixed-width enemy slot observation layout."""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.bridge.state_adapter import StateAdapter
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.characters.all import SUPPORTED_TRAINING_CHARACTERS, get_character
from sts2_env.core.combat import CombatState
from sts2_env.core.constants import MAX_ENEMIES
from sts2_env.core.rng import Rng
from sts2_env.gym_env.observation import (
    ENEMY_FEATURES,
    OBS_SIZE,
    RELIC_FEATURES,
    TOKEN_SLICES,
    encode_observation,
)
from sts2_env.monsters.act1_weak import create_fuzzy_wurm_crawler, create_shrinker_beetle


def _make_combat_with_n_enemies(n_enemies: int) -> CombatState:
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
    factories = [create_shrinker_beetle, create_fuzzy_wurm_crawler]
    for i in range(n_enemies):
        creature, ai = factories[i % len(factories)](rng)
        combat.add_enemy(creature, ai)
    combat.start_combat()
    return combat


def _minimal_bridge_state(*, n_enemies: int) -> dict:
    char_cfg = get_character("Ironclad")
    enemies = [
        {
            "id": f"ENEMY_{i}",
            "hp": 40,
            "max_hp": 50,
            "block": 0,
            "is_alive": True,
            "intent": "ATTACK",
            "intent_damage": 6,
            "intent_hits": 1,
            "powers": [],
        }
        for i in range(n_enemies)
    ]
    return {
        "type": "combat_action",
        "player": {
            "hp": char_cfg.starting_hp,
            "max_hp": char_cfg.starting_hp,
            "block": 0,
            "energy": 3,
            "max_energy": 3,
            "powers": [],
            "character_id": "Ironclad",
            "stars": 0,
            "orb_queue": {"capacity": 0, "count": 0, "orbs": []},
            "osty": {"alive": False, "hp": 0, "max_hp": 0, "block": 0},
        },
        "hand": [],
        "enemies": enemies,
        "draw_pile_count": 0,
        "discard_pile_count": 0,
        "exhaust_pile_count": 0,
        "relics": [{"id": char_cfg.starting_relic.name, "rarity": "STARTER", "enabled": True}],
    }


def _assert_enemy_slot_layout(obs: np.ndarray, n_enemies: int) -> None:
    enemy_start, enemy_end = TOKEN_SLICES["enemies"]
    mech_start, mech_end = TOKEN_SLICES["mechanics"]
    relic_start, _ = TOKEN_SLICES["relics"]

    assert obs.shape == (OBS_SIZE,)
    assert enemy_end == mech_start

    for slot in range(MAX_ENEMIES):
        slot_start = enemy_start + slot * ENEMY_FEATURES
        slot_slice = obs[slot_start:slot_start + ENEMY_FEATURES]
        if slot < n_enemies:
            assert obs[slot_start] == 1.0
            assert slot_slice.sum() > 0.0
        else:
            assert slot_slice.sum() == pytest.approx(0.0)

    if n_enemies < MAX_ENEMIES:
        unused_start = enemy_start + n_enemies * ENEMY_FEATURES
        assert obs[unused_start:mech_start].sum() == pytest.approx(0.0)

    mechanics = obs[mech_start:mech_end]
    char_one_hot = mechanics[: len(SUPPORTED_TRAINING_CHARACTERS)]
    assert char_one_hot.sum() == pytest.approx(1.0)
    assert char_one_hot[0] == 1.0

    assert obs[relic_start:relic_start + RELIC_FEATURES].sum() > 0.0


@pytest.mark.parametrize("n_enemies", [0, 1, 2])
def test_simulator_enemy_slots_preserve_token_slices(n_enemies: int) -> None:
    combat = _make_combat_with_n_enemies(n_enemies)
    assert len(combat.enemies) == n_enemies
    _assert_enemy_slot_layout(encode_observation(combat), n_enemies)


@pytest.mark.parametrize("n_enemies", [0, 1, 2])
def test_bridge_enemy_slots_preserve_token_slices(n_enemies: int) -> None:
    adapter = StateAdapter()
    _assert_enemy_slot_layout(
        adapter.encode_observation(_minimal_bridge_state(n_enemies=n_enemies)),
        n_enemies,
    )

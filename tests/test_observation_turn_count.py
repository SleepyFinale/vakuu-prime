"""Tests for normalized turn counter in combat observation (obs v11)."""

from __future__ import annotations

import pytest

from sts2_env.bridge.state_adapter import StateAdapter
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.characters.all import get_character
from sts2_env.core.combat import CombatState
from sts2_env.gym_env.observation import (
    OBS_TURN_COUNT_CAP,
    PLAYER_CORE_FEATURES,
    encode_observation,
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


def test_turn_count_after_start_combat() -> None:
    combat = _make_combat()
    obs = encode_observation(combat)
    assert obs[5] == pytest.approx(1 / OBS_TURN_COUNT_CAP)
    assert obs[PLAYER_CORE_FEATURES] == pytest.approx(0.0)


def test_turn_count_mid_combat() -> None:
    combat = _make_combat()
    combat.turn_count = 10
    obs = encode_observation(combat)
    assert obs[5] == pytest.approx(10 / OBS_TURN_COUNT_CAP)


def test_turn_count_capped_at_20() -> None:
    combat = _make_combat()
    combat.turn_count = 25
    obs = encode_observation(combat)
    assert obs[5] == pytest.approx(1.0)


def test_state_adapter_encodes_round_as_turn_count() -> None:
    adapter = StateAdapter()
    state = {
        "round": 10,
        "player": {
            "hp": 70,
            "max_hp": 80,
            "block": 0,
            "energy": 3,
            "max_energy": 3,
            "powers": [],
        },
        "hand": [],
        "enemies": [],
        "draw_pile": [],
        "discard_pile": [],
        "exhaust_pile": [],
    }
    obs = adapter.encode_observation(state)
    assert obs[5] == pytest.approx(10 / OBS_TURN_COUNT_CAP)

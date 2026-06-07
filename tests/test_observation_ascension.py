"""Tests for ascension level in combat observation (obs v10)."""

from __future__ import annotations

import pytest

from sts2_env.bridge.state_adapter import StateAdapter
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.characters.all import get_character
from sts2_env.core.combat import CombatState
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.gym_env.observation import (
    OBS_ASCENSION_SCALE,
    PLAYER_CORE_FEATURES,
    encode_observation,
)


def _make_combat(ascension_level: int = 0) -> CombatState:
    char_cfg = get_character("Ironclad")
    combat = CombatState(
        player_hp=char_cfg.starting_hp,
        player_max_hp=char_cfg.starting_hp,
        deck=create_ironclad_starter_deck(),
        rng_seed=42,
        character_id="Ironclad",
        relics=[char_cfg.starting_relic],
        ascension_level=ascension_level,
    )
    combat.start_combat()
    return combat


def test_ascension_encoded_in_player_state() -> None:
    combat = _make_combat(ascension_level=5)
    obs = encode_observation(combat)
    assert obs[4] == pytest.approx(5 / OBS_ASCENSION_SCALE)
    assert obs[PLAYER_CORE_FEATURES] == pytest.approx(0.0)


def test_ascension_defaults_to_zero() -> None:
    combat = _make_combat()
    obs = encode_observation(combat)
    assert obs[4] == pytest.approx(0.0)


def test_state_adapter_encodes_ascension_from_json() -> None:
    adapter = StateAdapter()
    state = {
        "ascension_level": 8,
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
    assert obs[4] == pytest.approx(8 / OBS_ASCENSION_SCALE)


def test_combat_env_passes_ascension_level() -> None:
    env = STS2CombatEnv(ascension_level=10)
    obs, _ = env.reset(seed=42)
    assert obs[4] == pytest.approx(10 / OBS_ASCENSION_SCALE)

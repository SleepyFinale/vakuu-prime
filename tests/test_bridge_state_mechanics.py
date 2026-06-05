"""Tests for bridge character-mechanics observation encoding."""

import numpy as np
import pytest

from sts2_env.bridge.state_adapter import StateAdapter
from sts2_env.cards.defect import create_defect_starter_deck
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.characters.all import get_character
from sts2_env.core.combat import CombatState
from sts2_env.gym_env.observation import BASE_OBS_SIZE, OBS_SIZE, encode_observation
from sts2_env.parity.bridge_replay import combat_state_to_bridge_state


def _minimal_combat_state(
    *,
    character_id: str,
    hand: list | None = None,
    enemies: list | None = None,
) -> dict:
    char_cfg = get_character(character_id)
    return {
        "type": "combat_action",
        "player": {
            "hp": char_cfg.starting_hp,
            "max_hp": char_cfg.starting_hp,
            "block": 0,
            "energy": 3,
            "max_energy": 3,
            "powers": [],
            "character_id": character_id,
            "stars": 0,
            "orb_queue": {"capacity": 0, "count": 0, "orbs": []},
            "osty": {"alive": False, "hp": 0, "max_hp": 0, "block": 0},
        },
        "hand": hand or [],
        "enemies": enemies or [],
        "draw_pile_count": 0,
        "discard_pile_count": 0,
        "exhaust_pile_count": 0,
    }


def test_ironclad_bridge_mechanics_one_hot_only():
    adapter = StateAdapter()
    obs = adapter.encode_observation(_minimal_combat_state(character_id="Ironclad"))
    assert obs.shape == (OBS_SIZE,)
    mechanics = obs[BASE_OBS_SIZE:]
    assert mechanics[0] == 1.0
    assert mechanics[1:].sum() == pytest.approx(0.0)


def test_defect_bridge_orb_mechanics():
    adapter = StateAdapter()
    state = _minimal_combat_state(character_id="Defect")
    state["player"]["orb_queue"] = {
        "capacity": 3,
        "count": 1,
        "orbs": [{"type": "LIGHTNING", "evoke_value": 8}],
    }
    obs = adapter.encode_observation(state)
    orb_count = obs[BASE_OBS_SIZE + 5 + 1 + 1]
    assert orb_count > 0


def test_regent_bridge_stars_mechanics():
    adapter = StateAdapter()
    state = _minimal_combat_state(character_id="Regent")
    state["player"]["stars"] = 6
    obs = adapter.encode_observation(state)
    stars_value = obs[BASE_OBS_SIZE + 5]
    assert stars_value == pytest.approx(6 / 30.0)


def test_necrobinder_bridge_osty_mechanics():
    adapter = StateAdapter()
    state = _minimal_combat_state(character_id="Necrobinder")
    state["player"]["osty"] = {
        "alive": True,
        "hp": 4,
        "max_hp": 5,
        "block": 2,
    }
    obs = adapter.encode_observation(state)
    osty_start = BASE_OBS_SIZE + 17 - 3
    assert obs[osty_start] == 1.0
    assert obs[osty_start + 1] == pytest.approx(4 / 5)


def _make_combat(deck, character_id: str) -> CombatState:
    char_cfg = get_character(character_id)
    combat = CombatState(
        player_hp=char_cfg.starting_hp,
        player_max_hp=char_cfg.starting_hp,
        deck=deck,
        rng_seed=123,
        character_id=character_id,
        relics=[char_cfg.starting_relic],
    )
    combat.start_combat()
    return combat


def test_bridge_round_trip_matches_simulator_observation():
    combat = _make_combat(create_ironclad_starter_deck(), "Ironclad")
    adapter = StateAdapter()
    sim_obs = encode_observation(combat)
    bridge_obs = adapter.encode_observation(combat_state_to_bridge_state(combat))
    np.testing.assert_allclose(sim_obs, bridge_obs, rtol=0, atol=1e-5)


def test_defect_orb_round_trip_matches_simulator_observation():
    combat = _make_combat(create_defect_starter_deck(), "Defect")
    combat.channel_orb(combat.player, "LIGHTNING")
    adapter = StateAdapter()
    sim_obs = encode_observation(combat)
    bridge_obs = adapter.encode_observation(combat_state_to_bridge_state(combat))
    np.testing.assert_allclose(sim_obs, bridge_obs, rtol=0, atol=1e-5)

"""Focused tests for bridge-side potion action masking and decoding."""

from __future__ import annotations

from sts2_env.bridge.state_adapter import StateAdapter
from sts2_env.core.constants import POTION_ACTION_START, POTION_TARGET_OPTIONS
from sts2_env.gym_env.observation import COMBAT_OBS_V2_SIZE, OBS_SIZE, RELIC_FEATURES


def _base_state() -> dict:
    return {
        "type": "combat_action",
        "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3, "max_energy": 3, "powers": []},
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
                "powers": [],
            }
        ],
        "draw_pile_count": 5,
        "discard_pile_count": 2,
        "exhaust_pile_count": 1,
    }


def test_encode_observation_encodes_relic_slots() -> None:
    adapter = StateAdapter()
    state = _base_state()
    state["relics"] = [
        {"id": "BurningBlood", "rarity": "STARTER", "enabled": True, "used_up": False},
        {"id": "Vajra", "rarity": "COMMON", "enabled": True, "used_up": False},
    ]
    obs = adapter.encode_observation(state)

    assert obs.shape == (OBS_SIZE,)
    relic_slice = obs[COMBAT_OBS_V2_SIZE:OBS_SIZE].reshape(-1, RELIC_FEATURES)
    assert relic_slice[0].sum() > 0
    assert relic_slice[1].sum() > 0
    assert relic_slice[2:].sum() == 0.0


def test_encode_observation_encodes_pile_memory_when_card_lists_present() -> None:
    adapter = StateAdapter()
    state = _base_state()
    state["draw_pile"] = [{"id": "Strike", "cost": 1, "type": "Attack"}]
    state["discard_pile"] = [{"id": "Defend", "cost": 1, "type": "Skill"}]
    state["play_pile"] = []
    obs = adapter.encode_observation(state)

    pile_start = 4 + 6 + 50
    assert obs[pile_start] == 1 / 20.0
    assert obs[pile_start + 1] == 1 / 20.0
    assert obs[pile_start + 2] == 1 / 20.0
    memory = obs[pile_start + 3:pile_start + 3 + 26]
    assert memory.sum() > 0.0
    assert obs[pile_start + 29:pile_start + 32].sum() == 0.0


def test_compute_action_mask_includes_targeted_and_untargeted_potions() -> None:
    adapter = StateAdapter()
    state = _base_state()
    state["potions"] = [
        {"slot": 0, "id": "BlockPotion", "usage": "CombatOnly", "target": "Self", "can_use": True},
        {
            "slot": 1,
            "id": "FirePotion",
            "usage": "AnyTime",
            "target": "AnyEnemy",
            "requires_target": True,
            "can_use": True,
        },
        {"slot": 2, "id": "FairyInABottle", "usage": "Automatic", "target": "Self", "can_use": True},
    ]

    mask = adapter.compute_action_mask(state)

    assert mask[POTION_ACTION_START] == 1
    fire_base = POTION_ACTION_START + POTION_TARGET_OPTIONS
    assert mask[fire_base] == 0
    assert mask[fire_base + 1] == 1
    fairy_base = POTION_ACTION_START + 2 * POTION_TARGET_OPTIONS
    assert mask[fairy_base] == 0


def test_decode_action_returns_out_of_hand_play_payload_for_potion() -> None:
    adapter = StateAdapter()
    decoded = adapter.decode_action(POTION_ACTION_START + POTION_TARGET_OPTIONS + 1, _base_state())

    assert decoded == {
        "type": "PLAY",
        "out_of_hand": True,
        "potion_slot": 1,
        "target_index": 0,
    }

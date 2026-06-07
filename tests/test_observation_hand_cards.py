"""Tests for expanded hand-card observation encoding (obs v7)."""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.bridge.state_adapter import StateAdapter
from sts2_env.cards.ironclad import (
    make_inflame,
    make_molten_fist,
    make_sword_boomerang,
)
from sts2_env.cards.silent import make_deflect, make_snakebite
from sts2_env.core.combat import CombatState
from sts2_env.core.constants import MAX_HAND_SIZE
from sts2_env.core.enums import CardId
from sts2_env.gym_env.observation import (
    CARD_FEATURES,
    EMPTY_HAND_SLOT_COST,
    NUM_PLAYER_POWERS,
    OBS_SIZE,
    PLAYER_CORE_FEATURES,
    TOKEN_SLICES,
    encode_card_features,
    encode_card_features_from_fields,
    encode_observation,
)

_HAND_START = PLAYER_CORE_FEATURES + NUM_PLAYER_POWERS


def _hand_slice(obs: np.ndarray) -> np.ndarray:
    hand_start, hand_end = TOKEN_SLICES["hand"]
    assert hand_end - hand_start == MAX_HAND_SIZE * CARD_FEATURES
    return obs[hand_start:hand_end].reshape(MAX_HAND_SIZE, CARD_FEATURES)


def _minimal_combat(hand_cards: list) -> CombatState:
    combat = CombatState(player_hp=70, player_max_hp=80, deck=[], rng_seed=1)
    combat.hand = hand_cards
    return combat


def test_obs_v7_size_constants() -> None:
    assert CARD_FEATURES == 9
    assert OBS_SIZE == 1985
    hand_start, hand_end = TOKEN_SLICES["hand"]
    assert hand_end - hand_start == 90
    assert hand_start == _HAND_START


def test_encode_card_features_extended_fields() -> None:
    molten = make_molten_fist()
    snakebite = make_snakebite()
    boomerang = make_sword_boomerang()
    inflame = make_inflame()

    assert encode_card_features(molten)[6] == pytest.approx(1.0)
    assert encode_card_features(molten)[7] == pytest.approx(0.0)

    assert encode_card_features(snakebite)[7] == pytest.approx(1.0)
    assert encode_card_features(snakebite)[6] == pytest.approx(0.0)

    assert encode_card_features(boomerang)[8] == pytest.approx(3 / 5.0)
    assert encode_card_features(inflame)[5] == pytest.approx(1.0)
    assert encode_card_features(inflame)[4] == pytest.approx(0.0)


def test_encode_observation_hand_slots_and_empty_padding() -> None:
    cards = [make_molten_fist(), make_sword_boomerang()]
    obs = encode_observation(_minimal_combat(cards))
    assert obs.shape == (OBS_SIZE,)

    hand = _hand_slice(obs)
    assert hand[0, 6] == pytest.approx(1.0)
    assert hand[1, 8] == pytest.approx(3 / 5.0)
    assert np.all(hand[2:, 0] == 0.0)
    assert np.all(hand[2:, 1] == pytest.approx(EMPTY_HAND_SLOT_COST))
    assert np.all(hand[2:, 2:] == 0.0)


def test_empty_slot_distinguishable_from_zero_cost_card() -> None:
    obs = encode_observation(_minimal_combat([make_deflect()]))
    hand = _hand_slice(obs)
    assert hand[0, 0] > 0.0
    assert hand[0, 1] == pytest.approx(0.0)
    assert hand[1, 0] == pytest.approx(0.0)
    assert hand[1, 1] == pytest.approx(EMPTY_HAND_SLOT_COST)


def test_bridge_empty_hand_slots_match_simulator_sentinel() -> None:
    adapter = StateAdapter()
    state = {
        "type": "combat_action",
        "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3, "max_energy": 3, "powers": []},
        "hand": [{"id": CardId.DEFLECT.name, "cost": 0, "type": "Skill", "base_block": 4}],
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
        "draw_pile_count": 0,
        "discard_pile_count": 0,
        "exhaust_pile_count": 0,
    }
    bridge_hand = _hand_slice(adapter.encode_observation(state))
    sim_hand = _hand_slice(encode_observation(_minimal_combat([make_deflect()])))
    np.testing.assert_allclose(bridge_hand, sim_hand, rtol=1e-5, atol=1e-5)


def test_bridge_adapter_matches_simulator_for_rich_hand_json() -> None:
    adapter = StateAdapter()
    state = {
        "type": "combat_action",
        "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3, "max_energy": 3, "powers": []},
        "hand": [
            {
                "id": "MOLTEN_FIST",
                "cost": 1,
                "type": "Attack",
                "base_damage": 10,
                "keywords": ["exhaust"],
                "hit_count": 1,
            },
            {
                "id": "SWORD_BOOMERANG",
                "cost": 1,
                "type": "Attack",
                "base_damage": 3,
                "hit_count": 3,
            },
            {
                "id": "SNAKEBITE",
                "cost": 2,
                "type": "Skill",
                "keywords": ["retain"],
                "retain": True,
            },
            {
                "id": "INFLAME",
                "cost": 1,
                "type": "Power",
            },
        ],
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
        "draw_pile_count": 0,
        "discard_pile_count": 0,
        "exhaust_pile_count": 0,
    }

    bridge_obs = adapter.encode_observation(state)
    sim_obs = encode_observation(
        _minimal_combat([make_molten_fist(), make_sword_boomerang(), make_snakebite(), make_inflame()])
    )

    bridge_hand = _hand_slice(bridge_obs)
    sim_hand = _hand_slice(sim_obs)
    np.testing.assert_allclose(bridge_hand[:4], sim_hand[:4], rtol=1e-5, atol=1e-5)


def test_bridge_fallback_fills_missing_keywords_from_reference_card() -> None:
    features = encode_card_features_from_fields(
        card_id="MOLTEN_FIST",
        cost=1,
        card_type="Attack",
    )
    assert features[6] == pytest.approx(1.0)
    assert features[2] == pytest.approx(10 / 50.0)

    boomerang = encode_card_features_from_fields(
        card_id="SWORD_BOOMERANG",
        cost=1,
        card_type="Attack",
    )
    assert boomerang[8] == pytest.approx(3 / 5.0)


def test_bridge_adapter_fallback_hand_without_keywords() -> None:
    adapter = StateAdapter()
    state = {
        "type": "combat_action",
        "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3, "max_energy": 3, "powers": []},
        "hand": [{"id": CardId.MOLTEN_FIST.name, "cost": 1, "type": "Attack"}],
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
        "draw_pile_count": 0,
        "discard_pile_count": 0,
        "exhaust_pile_count": 0,
    }
    hand = _hand_slice(adapter.encode_observation(state))
    assert hand[0, 6] == pytest.approx(1.0)

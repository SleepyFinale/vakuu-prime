"""Tests for draw-pile memory encoding (obs v4)."""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.bridge.state_adapter import StateAdapter
from sts2_env.cards.ironclad_basic import make_bash, make_defend_ironclad, make_strike_ironclad
from sts2_env.core.enums import CardId, CardType
from sts2_env.gym_env.observation import OBS_SIZE, PILE_FEATURES, encode_observation
from sts2_env.gym_env.pile_distribution import (
    PILE_COUNT_FEATURES,
    PILE_MEMORY_FEATURES,
    SimplePileCard,
    cards_from_combat,
    encode_pile_memory,
    encode_pile_summaries,
    hypergeom_at_least_one,
    hypergeom_expected_count,
)


def _pile_start_index() -> int:
    return 4 + 6 + 50


def _pile_memory_slice(obs: np.ndarray) -> np.ndarray:
    start = _pile_start_index() + PILE_COUNT_FEATURES
    return obs[start:start + PILE_MEMORY_FEATURES]


def test_hypergeom_at_least_one() -> None:
    assert hypergeom_at_least_one(10, 5, 0) == 0.0
    assert hypergeom_at_least_one(10, 5, 10) == 1.0
    assert hypergeom_at_least_one(0, 5, 3) == 0.0
    assert hypergeom_at_least_one(10, 10, 3) == 1.0
    assert 0.0 < hypergeom_at_least_one(10, 5, 3) < 1.0


def test_hypergeom_expected_count() -> None:
    assert hypergeom_expected_count(10, 5, 4) == pytest.approx(2.0)
    assert hypergeom_expected_count(10, 0, 4) == 0.0


def test_known_draw_order_reflects_top_of_draw_pile() -> None:
    draw = [
        SimplePileCard(CardId.DEFEND_IRONCLAD, CardType.SKILL),
        SimplePileCard(CardId.STRIKE_IRONCLAD, CardType.ATTACK),
    ]
    memory = encode_pile_memory(draw, [], [], [], next_draw_count=5)

    assert memory[10] == pytest.approx(0.66)
    assert memory[11] == pytest.approx(1.0)
    assert memory[12:15].sum() == pytest.approx(0.0)


def test_empty_draw_with_attack_discard_sets_high_attack_probability() -> None:
    attacks = [
        SimplePileCard(CardId.STRIKE_IRONCLAD, CardType.ATTACK),
        SimplePileCard(CardId.STRIKE_IRONCLAD, CardType.ATTACK),
    ]
    memory = encode_pile_memory([], attacks, [], [], next_draw_count=5)

    assert memory[0] == pytest.approx(1.0)
    assert memory[5] == pytest.approx(1.0)
    assert memory[15] == pytest.approx(1.0)


def test_hand_block_discard_attacks_visible_in_unseen_composition(simple_combat) -> None:
    combat = simple_combat
    combat.hand = [make_defend_ironclad(), make_defend_ironclad()]
    combat.draw_pile = [make_strike_ironclad()]
    combat.discard_pile = [make_bash()]
    combat.play_pile = []
    combat.exhaust_pile = []

    memory = encode_pile_memory(*cards_from_combat(combat), next_draw_count=5)

    assert memory[0] > 0.5
    assert memory[1] < 0.5


def test_encode_observation_includes_pile_memory(simple_combat) -> None:
    combat = simple_combat
    combat.draw_pile = [make_strike_ironclad(), make_defend_ironclad()]
    combat.discard_pile = [make_bash()]
    combat.hand = [make_defend_ironclad()]
    combat.play_pile = []

    obs = encode_observation(combat)
    assert obs.shape == (OBS_SIZE,)
    memory = _pile_memory_slice(obs)
    assert memory.sum() > 0.0
    assert obs[_pile_start_index() + PILE_FEATURES - 3: _pile_start_index() + PILE_FEATURES].sum() == 0.0


def test_bridge_adapter_encodes_pile_memory_from_card_lists() -> None:
    adapter = StateAdapter()
    state = {
        "type": "combat_action",
        "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3, "max_energy": 3, "powers": []},
        "hand": [{"id": "Defend", "cost": 1, "type": "Skill"}],
        "draw_pile": [{"id": "Strike", "cost": 1, "type": "Attack"}],
        "discard_pile": [{"id": "Bash", "cost": 2, "type": "Attack"}],
        "play_pile": [],
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
        "draw_pile_count": 1,
        "discard_pile_count": 1,
        "exhaust_pile_count": 0,
    }
    obs = adapter.encode_observation(state)

    assert obs.shape == (OBS_SIZE,)
    memory = _pile_memory_slice(obs)
    assert memory[0] > 0.5
    assert memory[5] > 0.0


def test_encode_pile_summaries_shape() -> None:
    summaries = encode_pile_summaries([], [], [], [], 0)
    assert summaries.shape == (PILE_FEATURES,)

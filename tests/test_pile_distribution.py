"""Tests for draw-pile memory encoding (obs v4)."""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.bridge.state_adapter import StateAdapter
from sts2_env.cards.ironclad_basic import make_bash, make_defend_ironclad, make_strike_ironclad
from sts2_env.core.enums import CardId, CardType
from sts2_env.gym_env.observation import (
    CARD_FEATURES,
    CARD_IDS,
    MAX_HAND_SIZE,
    NUM_CARD_IDS,
    NUM_PLAYER_POWERS,
    OBS_SIZE,
    PILE_FEATURES,
    PLAYER_CORE_FEATURES,
    encode_observation,
)
from sts2_env.gym_env.pile_distribution import (
    PILE_COUNT_FEATURES,
    PILE_MEMORY_FEATURES,
    SimplePileCard,
    cards_from_combat,
    clear_watchlist_cache,
    encode_pile_memory,
    encode_pile_summaries,
    hypergeom_at_least_one,
    hypergeom_expected_count,
    load_watchlist_groups,
)
from sts2_env.gym_env.pile_distribution import (
    _expected_type_draws,
    _prob_at_least_one_type,
)


def _pile_start_index() -> int:
    return PLAYER_CORE_FEATURES + NUM_PLAYER_POWERS + MAX_HAND_SIZE * CARD_FEATURES


def _card_id_norm(card_id: CardId) -> float:
    index = next(i for i, cid in enumerate(CARD_IDS) if cid == card_id)
    return (index + 1) / (NUM_CARD_IDS + 1)


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


def test_next_draw_features_match_helper_functions() -> None:
    """Regression: batched type counts must match unbatched helper semantics."""
    draw = [
        SimplePileCard(CardId.STRIKE_IRONCLAD, CardType.ATTACK),
        SimplePileCard(CardId.DEFEND_IRONCLAD, CardType.SKILL),
        SimplePileCard(CardId.BASH, CardType.ATTACK),
    ]
    discard = [
        SimplePileCard(CardId.INFLAME, CardType.POWER),
        SimplePileCard(CardId.ARMAMENTS, CardType.SKILL),
    ]
    play = [SimplePileCard(CardId.STRIKE_IRONCLAD, CardType.ATTACK)]
    hand = [SimplePileCard(CardId.DEFEND_IRONCLAD, CardType.SKILL, should_retain=True)]
    next_draw_count = 4

    memory = encode_pile_memory(draw, discard, play, hand, next_draw_count=next_draw_count)

    known_count = min(next_draw_count, len(draw))
    known_cards = list(draw[:known_count])
    remaining_draws = max(0, next_draw_count - known_count)
    pending_discard = list(play) + [card for card in hand if not card.should_retain]
    shuffle_pool = list(draw[known_count:]) + list(discard) + pending_discard

    expected = [
        _prob_at_least_one_type(
            known_cards, shuffle_pool, remaining_draws=remaining_draws, card_type=CardType.ATTACK,
        ),
        _prob_at_least_one_type(
            known_cards, shuffle_pool, remaining_draws=remaining_draws, card_type=CardType.SKILL,
        ),
        _prob_at_least_one_type(
            known_cards, shuffle_pool, remaining_draws=remaining_draws, card_type=CardType.POWER,
        ),
        _expected_type_draws(
            known_cards, shuffle_pool, remaining_draws=remaining_draws, card_type=CardType.ATTACK,
        ),
        _expected_type_draws(
            known_cards, shuffle_pool, remaining_draws=remaining_draws, card_type=CardType.SKILL,
        ),
    ]

    assert memory[5:10] == pytest.approx(expected)


def test_known_draw_order_reflects_top_of_draw_pile() -> None:
    draw = [
        SimplePileCard(CardId.DEFEND_IRONCLAD, CardType.SKILL),
        SimplePileCard(CardId.STRIKE_IRONCLAD, CardType.ATTACK),
    ]
    memory = encode_pile_memory(draw, [], [], [], next_draw_count=5)

    assert memory[10] == pytest.approx(_card_id_norm(CardId.DEFEND_IRONCLAD))
    assert memory[11] == pytest.approx(0.66)
    assert memory[12] == pytest.approx(_card_id_norm(CardId.STRIKE_IRONCLAD))
    assert memory[13] == pytest.approx(1.0)
    assert memory[14:20].sum() == pytest.approx(0.0)


def test_known_draw_order_distinguishes_card_ids_with_same_type() -> None:
    defend_memory = encode_pile_memory(
        [SimplePileCard(CardId.DEFEND_IRONCLAD, CardType.SKILL)],
        [], [], [],
        next_draw_count=1,
    )
    armament_memory = encode_pile_memory(
        [SimplePileCard(CardId.ARMAMENTS, CardType.SKILL)],
        [], [], [],
        next_draw_count=1,
    )

    assert defend_memory[10] != armament_memory[10]
    assert defend_memory[11] == armament_memory[11] == pytest.approx(0.66)


def test_empty_draw_with_attack_discard_sets_high_attack_probability() -> None:
    attacks = [
        SimplePileCard(CardId.STRIKE_IRONCLAD, CardType.ATTACK),
        SimplePileCard(CardId.STRIKE_IRONCLAD, CardType.ATTACK),
    ]
    memory = encode_pile_memory([], attacks, [], [], next_draw_count=5)

    assert memory[0] == pytest.approx(1.0)
    assert memory[5] == pytest.approx(1.0)
    assert memory[20] == pytest.approx(1.0)


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


def test_load_watchlist_groups_roundtrip(tmp_path) -> None:
    import json

    payload = {
        "version": 1,
        "groups": {
            "power": {"cards": ["DEMON_FORM_CARD"], "auto": None, "exclude": []},
            "finisher": {"cards": ["BASH"], "auto": None, "exclude": []},
            "setup": {"cards": ["CLAW"], "auto": None, "exclude": []},
            "aoe": {"cards": ["BONE_SHARDS"], "auto": None, "exclude": []},
        },
    }
    watchlist_path = tmp_path / "PILE_WATCHLIST.json"
    watchlist_path.write_text(json.dumps(payload), encoding="utf-8")

    clear_watchlist_cache()
    groups = load_watchlist_groups(str(watchlist_path))

    assert groups["power"] == frozenset({CardId.DEMON_FORM_CARD})
    assert groups["finisher"] == frozenset({CardId.BASH})
    assert groups["setup"] == frozenset({CardId.CLAW})
    assert groups["aoe"] == frozenset({CardId.BONE_SHARDS})
    clear_watchlist_cache()


def test_watchlist_unknown_card_skipped(tmp_path) -> None:
    import json

    payload = {
        "version": 1,
        "groups": {
            "power": {"cards": ["NOT_A_REAL_CARD", "DEMON_FORM_CARD"], "auto": None, "exclude": []},
            "finisher": {"cards": [], "auto": None, "exclude": []},
            "setup": {"cards": [], "auto": None, "exclude": []},
            "aoe": {"cards": [], "auto": None, "exclude": []},
        },
    }
    watchlist_path = tmp_path / "PILE_WATCHLIST.json"
    watchlist_path.write_text(json.dumps(payload), encoding="utf-8")

    clear_watchlist_cache()
    groups = load_watchlist_groups(str(watchlist_path))

    assert groups["power"] == frozenset({CardId.DEMON_FORM_CARD})
    clear_watchlist_cache()


def test_encode_pile_memory_watchlist_presence() -> None:
    bash = SimplePileCard(CardId.BASH, CardType.ATTACK, base_damage=8)
    memory = encode_pile_memory([], [bash], [], [], next_draw_count=5)

    assert memory[28] == pytest.approx(1.0)

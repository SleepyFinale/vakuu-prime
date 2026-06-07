"""Tests for multi-intent enemy observation encoding."""

from __future__ import annotations

import pytest

from sts2_env.bridge.state_adapter import StateAdapter
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.characters.all import get_character
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import IntentType
from sts2_env.core.rng import Rng
from sts2_env.gym_env.observation import (
    ENEMY_CORE_FEATURES,
    ENEMY_FEATURES,
    INTENT_DAMAGE_SCALE,
    INTENT_HITS_CAP,
    INTENT_TYPES,
    NUM_INTENT_TYPES,
    TOKEN_SLICES,
    encode_observation,
)
from sts2_env.monsters.act1_weak import create_fuzzy_wurm_crawler
from sts2_env.monsters.intents import (
    Intent,
    attack_intent,
    buff_intent,
    debuff_intent,
    multi_attack_intent,
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
    rng = Rng(42)
    creature, ai = create_fuzzy_wurm_crawler(rng)
    combat.add_enemy(creature, ai)
    combat.start_combat()
    return combat


def _intent_slice(obs, enemy_slot: int = 0):
    enemy_start = TOKEN_SLICES["enemies"][0]
    slot_start = enemy_start + enemy_slot * ENEMY_FEATURES
    intent_start = slot_start + 3
    return obs[intent_start:intent_start + ENEMY_CORE_FEATURES - 3]


def _set_enemy_intents(combat: CombatState, intents: list[Intent]) -> None:
    enemy = combat.enemies[0]
    ai = combat.enemy_ais[enemy.combat_id]
    ai.current_move.intents = intents


def _intent_type_index(intent_type: IntentType) -> int:
    return INTENT_TYPES.index(intent_type)


def test_single_attack_intent() -> None:
    combat = _make_combat()
    _set_enemy_intents(combat, [attack_intent(12)])

    intent_features = _intent_slice(encode_observation(combat))
    assert intent_features[_intent_type_index(IntentType.ATTACK)] == 1.0
    assert intent_features[NUM_INTENT_TYPES] == pytest.approx(12 / INTENT_DAMAGE_SCALE)
    assert intent_features[NUM_INTENT_TYPES + 1] == pytest.approx(1 / INTENT_HITS_CAP)


def test_multi_attack_intent() -> None:
    combat = _make_combat()
    _set_enemy_intents(combat, [multi_attack_intent(5, 3)])

    intent_features = _intent_slice(encode_observation(combat))
    assert intent_features[_intent_type_index(IntentType.MULTI_ATTACK)] == 1.0
    assert intent_features[NUM_INTENT_TYPES] == pytest.approx(15 / INTENT_DAMAGE_SCALE)
    assert intent_features[NUM_INTENT_TYPES + 1] == pytest.approx(3 / INTENT_HITS_CAP)


def test_attack_then_buff_sets_both_bits() -> None:
    combat = _make_combat()
    _set_enemy_intents(combat, [attack_intent(10), buff_intent()])

    intent_features = _intent_slice(encode_observation(combat))
    assert intent_features[_intent_type_index(IntentType.ATTACK)] == 1.0
    assert intent_features[_intent_type_index(IntentType.BUFF)] == 1.0
    assert intent_features[NUM_INTENT_TYPES] == pytest.approx(10 / INTENT_DAMAGE_SCALE)
    assert intent_features[NUM_INTENT_TYPES + 1] == pytest.approx(1 / INTENT_HITS_CAP)


def test_buff_then_attack_is_order_independent() -> None:
    combat = _make_combat()
    _set_enemy_intents(combat, [buff_intent(), attack_intent(10)])

    intent_features = _intent_slice(encode_observation(combat))
    assert intent_features[_intent_type_index(IntentType.ATTACK)] == 1.0
    assert intent_features[_intent_type_index(IntentType.BUFF)] == 1.0
    assert intent_features[NUM_INTENT_TYPES] == pytest.approx(10 / INTENT_DAMAGE_SCALE)
    assert intent_features[NUM_INTENT_TYPES + 1] == pytest.approx(1 / INTENT_HITS_CAP)


def test_attack_debuff_buff_sets_three_bits() -> None:
    combat = _make_combat()
    _set_enemy_intents(combat, [attack_intent(8), debuff_intent(), buff_intent()])

    intent_features = _intent_slice(encode_observation(combat))
    assert intent_features[_intent_type_index(IntentType.ATTACK)] == 1.0
    assert intent_features[_intent_type_index(IntentType.DEBUFF)] == 1.0
    assert intent_features[_intent_type_index(IntentType.BUFF)] == 1.0
    assert intent_features[NUM_INTENT_TYPES] == pytest.approx(8 / INTENT_DAMAGE_SCALE)


def test_hits_are_capped_at_ten() -> None:
    combat = _make_combat()
    _set_enemy_intents(combat, [multi_attack_intent(2, 12)])

    intent_features = _intent_slice(encode_observation(combat))
    assert intent_features[NUM_INTENT_TYPES] == pytest.approx(24 / INTENT_DAMAGE_SCALE)
    assert intent_features[NUM_INTENT_TYPES + 1] == pytest.approx(1.0)


def _bridge_state_for_intents(
    *,
    intent: str | None = "ATTACK",
    intent_types: list[str] | None = None,
    intent_damage: int = 0,
    intent_hits: int = 1,
) -> dict:
    char_cfg = get_character("Ironclad")
    enemy = {
        "id": "ENEMY_0",
        "hp": 40,
        "max_hp": 50,
        "block": 0,
        "is_alive": True,
        "intent_damage": intent_damage,
        "intent_hits": intent_hits,
        "powers": [],
    }
    if intent_types is not None:
        enemy["intent_types"] = intent_types
    if intent is not None:
        enemy["intent"] = intent
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
        "enemies": [enemy],
        "draw_pile_count": 0,
        "discard_pile_count": 0,
        "exhaust_pile_count": 0,
        "relics": [{"id": char_cfg.starting_relic.name, "rarity": "STARTER", "enabled": True}],
    }


def test_bridge_adapter_multi_intent_types() -> None:
    adapter = StateAdapter()
    state = _bridge_state_for_intents(
        intent_types=["ATTACK", "DEBUFF"],
        intent_damage=8,
        intent_hits=1,
    )
    intent_features = _intent_slice(adapter.encode_observation(state))
    assert intent_features[_intent_type_index(IntentType.ATTACK)] == 1.0
    assert intent_features[_intent_type_index(IntentType.DEBUFF)] == 1.0
    assert intent_features[NUM_INTENT_TYPES] == pytest.approx(8 / INTENT_DAMAGE_SCALE)
    assert intent_features[NUM_INTENT_TYPES + 1] == pytest.approx(1 / INTENT_HITS_CAP)


def test_bridge_adapter_single_intent_backward_compat() -> None:
    adapter = StateAdapter()
    state = _bridge_state_for_intents(
        intent="ATTACK",
        intent_damage=6,
        intent_hits=1,
    )
    intent_features = _intent_slice(adapter.encode_observation(state))
    assert intent_features[_intent_type_index(IntentType.ATTACK)] == 1.0
    assert intent_features[NUM_INTENT_TYPES] == pytest.approx(6 / INTENT_DAMAGE_SCALE)
    assert intent_features[NUM_INTENT_TYPES + 1] == pytest.approx(1 / INTENT_HITS_CAP)

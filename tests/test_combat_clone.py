"""Tests for CombatState cloning."""

from __future__ import annotations

import copy

import numpy as np

from sts2_env.cards.base import reset_instance_counter
from sts2_env.cards.factory import create_card
from sts2_env.cards.ironclad_basic import make_strike_ironclad
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import CardId, PowerId
from sts2_env.core.rng import Rng
from sts2_env.gym_env.observation import encode_observation
from sts2_env.monsters.act1_weak import create_shrinker_beetle
from sts2_env.search.combat_clone import _rewire_combat_pointers, clone_combat_state
from sts2_env.search.combat_step import apply_combat_action


def _clone_via_deepcopy(combat: CombatState) -> CombatState:
    cloned = copy.deepcopy(combat)
    _rewire_combat_pointers(cloned)
    return cloned


def test_clone_matches_deepcopy_observation(simple_combat):
    explicit = clone_combat_state(simple_combat)
    baseline = _clone_via_deepcopy(simple_combat)
    assert np.allclose(
        encode_observation(explicit),
        encode_observation(baseline),
    )


def test_clone_preserves_observation(simple_combat):
    cloned = clone_combat_state(simple_combat)
    obs1 = encode_observation(simple_combat)
    obs2 = encode_observation(cloned)
    assert np.allclose(obs1, obs2)


def test_clone_diverges_after_different_actions(simple_combat):
    cloned_a = clone_combat_state(simple_combat)
    cloned_b = clone_combat_state(simple_combat)

    mask_actions = __import__(
        "sts2_env.gym_env.action_space", fromlist=["get_action_mask"]
    ).get_action_mask(simple_combat)
    legal = np.flatnonzero(mask_actions)
    play_action = int(legal[1]) if len(legal) > 1 else int(legal[0])

    apply_combat_action(cloned_a, play_action)
    apply_combat_action(cloned_b, 0)

    obs_a = encode_observation(cloned_a)
    obs_b = encode_observation(cloned_b)
    assert not np.allclose(obs_a, obs_b)


def test_combat_state_clone_method(simple_combat):
    cloned = simple_combat.clone()
    assert np.allclose(
        encode_observation(simple_combat),
        encode_observation(cloned),
    )


def test_clone_after_same_action_matches_original():
    reset_instance_counter()
    from sts2_env.cards.ironclad_basic import create_ironclad_starter_deck
    from sts2_env.core.combat import CombatState

    deck = create_ironclad_starter_deck()
    combat = CombatState(player_hp=80, player_max_hp=80, deck=deck, rng_seed=42)
    creature, ai = create_shrinker_beetle(Rng(42))
    combat.add_enemy(creature, ai)
    combat.start_combat()

    cloned = clone_combat_state(combat)
    mask = __import__(
        "sts2_env.gym_env.action_space", fromlist=["get_action_mask"]
    ).get_action_mask(combat)
    action = int(np.flatnonzero(mask)[0])

    apply_combat_action(combat, action)
    apply_combat_action(cloned, action)

    assert np.allclose(
        encode_observation(combat),
        encode_observation(cloned),
    )


def test_clone_card_dedup_across_piles(simple_combat):
    strike = simple_combat.hand[0]
    simple_combat._played_cards_combat.append(strike)

    cloned = clone_combat_state(simple_combat)
    assert cloned.hand[0] is not strike
    assert cloned._played_cards_combat[0] is cloned.hand[0]
    assert cloned._played_cards_combat[0] is not strike


def test_clone_rng_counter_is_independent(simple_combat):
    simple_combat.rng.next_int(0, 10)
    counter = simple_combat.rng.counter

    cloned_a = clone_combat_state(simple_combat)
    cloned_b = clone_combat_state(simple_combat)
    assert cloned_a.rng.counter == counter
    assert cloned_b.rng.counter == counter

    cloned_a.rng.next_int(0, 10)
    cloned_b.rng.next_int(0, 10)
    cloned_b.rng.next_int(0, 10)

    assert cloned_a.rng.counter == counter + 1
    assert cloned_b.rng.counter == counter + 2
    assert simple_combat.rng.counter == counter


def test_clone_matches_deepcopy_mid_combat_with_powers(simple_combat):
    simple_combat.apply_power_to(simple_combat.player, PowerId.STRENGTH, 2)
    simple_combat._played_cards_combat.append(make_strike_ironclad())

    explicit = clone_combat_state(simple_combat)
    baseline = _clone_via_deepcopy(simple_combat)
    assert np.allclose(
        encode_observation(explicit),
        encode_observation(baseline),
    )

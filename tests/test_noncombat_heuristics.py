"""Tests for non-combat heuristic policy helpers."""

import pytest

from sts2_env.cards.factory import create_card
from sts2_env.cards.ironclad_basic import make_strike_ironclad
from sts2_env.core.enums import CardId, CardType
from sts2_env.gym_env.noncombat_heuristics import (
    NoncombatHeuristicConfig,
    card_reward_global_action,
    pick_boss_relic_index,
    pick_card_reward_index,
)
from sts2_env.gym_env.run_env import _CARD_RWD_START, _LAYOUT
from sts2_env.run.run_manager import RunManager


def test_pick_card_reward_prefers_power():
    mgr = RunManager(seed=10, character_id="Ironclad")
    mgr._phase = RunManager.PHASE_CARD_REWARD
    strike = make_strike_ironclad()
    strike.card_type = CardType.ATTACK
    power = create_card(CardId.INFLAME)
    power.card_type = CardType.POWER
    mgr._offered_cards = [strike, power]
    mgr._current_reward = None

    assert pick_card_reward_index(mgr) == 1


def test_pick_card_reward_skips_large_deck():
    mgr = RunManager(seed=11, character_id="Ironclad")
    mgr._phase = RunManager.PHASE_CARD_REWARD
    mgr._offered_cards = [make_strike_ironclad()]
    mgr.run_state.player.deck = [make_strike_ironclad() for _ in range(31)]

    assert pick_card_reward_index(mgr) is None


def test_pick_boss_relic_prefers_black_star():
    mgr = RunManager(seed=12, character_id="Ironclad")
    mgr._boss_relics = ["ECTOPLASM", "BLACK_STAR", "SOZU"]
    assert pick_boss_relic_index(mgr) == 1


def test_card_reward_global_action_skip_uses_slot_three():
    mgr = RunManager(seed=13, character_id="Ironclad")
    mgr._phase = RunManager.PHASE_CARD_REWARD
    mgr._offered_cards = [make_strike_ironclad()]
    mgr.run_state.player.deck = [make_strike_ironclad() for _ in range(31)]

    action = card_reward_global_action(mgr, _LAYOUT)
    assert action == _CARD_RWD_START + 3


def test_heuristic_config_toggles():
    config = NoncombatHeuristicConfig(card_reward=False)
    mgr = RunManager(seed=14, character_id="Ironclad")
    mgr._phase = RunManager.PHASE_CARD_REWARD
    mgr._offered_cards = [make_strike_ironclad()]

    from sts2_env.gym_env.noncombat_heuristics import heuristic_global_action

    assert heuristic_global_action(mgr, config) is None


def test_learned_mode_delegates_to_model():
    pytest.importorskip("torch")
    from sts2_env.gym_env.card_value import build_card_value_net
    from sts2_env.gym_env.noncombat_heuristics import (
        NoncombatHeuristicConfig,
        pick_card_reward_index,
    )

    mgr = RunManager(seed=15, character_id="Ironclad")
    mgr._phase = RunManager.PHASE_CARD_REWARD
    mgr._offered_cards = [make_strike_ironclad()]

    config = NoncombatHeuristicConfig(
        card_reward_mode="learned",
        card_value_model=build_card_value_net(),
    )
    assert pick_card_reward_index(mgr, config) is not None

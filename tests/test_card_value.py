"""Tests for card-value encoding and picking."""

import numpy as np
import pytest

from sts2_env.cards.factory import create_card
from sts2_env.cards.ironclad_basic import make_strike_ironclad
from sts2_env.core.enums import CardId, CardType
from sts2_env.gym_env.card_value import (
    CARD_FEATURE_SIZE,
    MAX_CARD_OPTIONS,
    RUN_CONTEXT_SIZE,
    SKIP_LABEL,
    build_card_value_net,
    encode_card_features,
    encode_card_reward_sample,
    encode_run_context,
    label_from_rules,
    pick_card_reward_index,
)
from sts2_env.gym_env.noncombat_heuristics import (
    NoncombatHeuristicConfig,
    pick_card_reward_index_rules,
)
from sts2_env.run.run_manager import RunManager


def test_encode_shapes():
    mgr = RunManager(seed=1, character_id="Ironclad")
    mgr._phase = RunManager.PHASE_CARD_REWARD
    strike = make_strike_ironclad()
    power = create_card(CardId.INFLAME)
    mgr._offered_cards = [strike, power]

    context, cards, mask, n = encode_card_reward_sample(mgr)
    assert context.shape == (RUN_CONTEXT_SIZE,)
    assert cards.shape == (MAX_CARD_OPTIONS, CARD_FEATURE_SIZE)
    assert mask.shape == (MAX_CARD_OPTIONS,)
    assert n == 2


def test_label_matches_rules():
    mgr = RunManager(seed=2, character_id="Ironclad")
    mgr._phase = RunManager.PHASE_CARD_REWARD
    strike = make_strike_ironclad()
    strike.card_type = CardType.ATTACK
    power = create_card(CardId.INFLAME)
    power.card_type = CardType.POWER
    mgr._offered_cards = [strike, power]

    assert label_from_rules(mgr) == pick_card_reward_index_rules(mgr)


def test_card_value_net_forward():
    torch = pytest.importorskip("torch")

    config = build_card_value_net()
    model = config
    context = torch.randn(4, RUN_CONTEXT_SIZE)
    cards = torch.randn(4, MAX_CARD_OPTIONS, CARD_FEATURE_SIZE)
    mask = torch.zeros(4, MAX_CARD_OPTIONS)
    mask[:, :3] = 1.0
    logits = model(context, cards, mask)
    assert logits.shape == (4, MAX_CARD_OPTIONS + 1)


def test_learned_pick_with_mock_model():
    torch = pytest.importorskip("torch")

    mgr = RunManager(seed=3, character_id="Ironclad")
    mgr._phase = RunManager.PHASE_CARD_REWARD
    strike = make_strike_ironclad()
    mgr._offered_cards = [strike]

    model = build_card_value_net()
    pick = pick_card_reward_index(mgr, model)
    assert pick in (0, None)


def test_heuristic_config_learned_mode():
    torch = pytest.importorskip("torch")

    mgr = RunManager(seed=4, character_id="Ironclad")
    mgr._phase = RunManager.PHASE_CARD_REWARD
    mgr._offered_cards = [make_strike_ironclad()]

    config = NoncombatHeuristicConfig(
        card_reward_mode="learned",
        card_value_model=build_card_value_net(),
    )
    from sts2_env.gym_env.noncombat_heuristics import pick_card_reward_index

    assert pick_card_reward_index(mgr, config) is not None

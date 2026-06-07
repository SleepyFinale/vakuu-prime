"""Tests for combat critic value estimation."""

from __future__ import annotations

import numpy as np
import pytest

from sts2_env.cards.ironclad_basic import make_strike_ironclad
from sts2_env.core.combat import CombatState
from sts2_env.encounters import act1
from sts2_env.gym_env.combat_value import (
    CombatValueConfig,
    build_combat_from_deck,
    clone_run_deck,
    estimate_deck_value,
    pick_card_by_combat_value,
    predict_combat_values,
    score_card_draft_options,
)
from sts2_env.gym_env.observation import OBS_SIZE
from sts2_env.gym_env.run_reward import (
    NavigatorRewardConfig,
    RunRewardSnapshot,
    compute_draft_value_shaping,
    compute_navigator_shaping,
)
from sts2_env.run.run_manager import RunManager


class _MockValuePolicy:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def predict_values(self, obs_tensor):
        return obs_tensor.sum(dim=1, keepdim=True) * self.scale


class _MockCombatModel:
    def __init__(self, scale: float = 1.0) -> None:
        self.policy = _MockValuePolicy(scale=scale)
        self.device = "cpu"


class TestCombatValueBootstrap:
    def test_build_combat_from_deck_starts_combat(self):
        mgr = RunManager(seed=42, character_id="Ironclad")
        deck = clone_run_deck(mgr)
        setup = act1.ELITE_ENCOUNTERS[0]
        combat = build_combat_from_deck(mgr, deck, setup, rng_seed=1234)
        assert isinstance(combat, CombatState)
        assert not combat.is_over
        assert len(combat.enemies) > 0

    def test_clone_deck_with_extra_card(self):
        mgr = RunManager(seed=42, character_id="Ironclad")
        base_size = len(mgr.run_state.player.deck)
        deck = clone_run_deck(mgr, extra_card=make_strike_ironclad())
        assert len(deck) == base_size + 1


class TestCombatValuePredict:
    def test_predict_combat_values_shape(self):
        model = _MockCombatModel()
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        values = predict_combat_values(model, obs)
        assert values.shape == (1,)

    def test_estimate_deck_value_with_mock(self):
        mgr = RunManager(seed=7, character_id="Ironclad")
        deck = clone_run_deck(mgr)
        model = _MockCombatModel(scale=0.01)
        config = CombatValueConfig(num_encounters=1, rng_seed=0)
        value = estimate_deck_value(mgr, deck, model, config=config)
        assert isinstance(value, float)


class TestCombatValueDraft:
    def test_score_card_draft_options_returns_deltas(self):
        mgr = RunManager(seed=11, character_id="Ironclad")
        mgr._phase = RunManager.PHASE_CARD_REWARD
        strike = make_strike_ironclad()
        mgr._offered_cards = [strike, make_strike_ironclad()]
        model = _MockCombatModel(scale=0.01)
        deltas, baseline = score_card_draft_options(
            mgr, model, config=CombatValueConfig(num_encounters=1),
        )
        assert len(deltas) == 2
        assert isinstance(baseline, float)

    def test_pick_card_by_combat_value_returns_index(self):
        mgr = RunManager(seed=13, character_id="Ironclad")
        mgr._phase = RunManager.PHASE_CARD_REWARD
        mgr._offered_cards = [make_strike_ironclad()]
        model = _MockCombatModel(scale=0.01)
        pick, deltas, baseline = pick_card_by_combat_value(
            mgr, model, config=CombatValueConfig(num_encounters=1, skip_threshold=-999),
        )
        assert pick == 0
        assert len(deltas) == 1


class TestNavigatorRewardShaping:
    def test_draft_value_shaping(self):
        config = NavigatorRewardConfig(draft_value_scale=0.1)
        assert compute_draft_value_shaping(0.5, config) == pytest.approx(0.05)

    def test_navigator_shaping_includes_draft_delta(self):
        config = NavigatorRewardConfig(draft_value_scale=0.1)
        snap = RunRewardSnapshot(
            total_floor=2, hp_ratio=1.0, max_hp=80,
            phase=RunManager.PHASE_MAP_CHOICE, combat_active=False,
        )
        reward = compute_navigator_shaping(snap, snap, config, draft_delta=0.2)
        assert reward == pytest.approx(0.02)

"""Tests for Navigator reward shaping with combat-value draft signals."""

import pytest

from sts2_env.gym_env.run_reward import (
    NavigatorRewardConfig,
    RunRewardSnapshot,
    compute_draft_value_shaping,
    compute_navigator_shaping,
)
from sts2_env.run.run_manager import RunManager


class TestNavigatorReward:
    def test_zero_draft_delta_is_no_op(self):
        config = NavigatorRewardConfig(draft_value_scale=0.1)
        snap = RunRewardSnapshot(
            total_floor=1, hp_ratio=1.0, max_hp=80,
            phase=RunManager.PHASE_MAP_CHOICE, combat_active=False,
        )
        assert compute_navigator_shaping(snap, snap, config, draft_delta=0.0) == 0.0

    def test_positive_draft_delta_adds_shaping(self):
        config = NavigatorRewardConfig(draft_value_scale=0.2)
        prev = RunRewardSnapshot(
            total_floor=3, hp_ratio=1.0, max_hp=80,
            phase=RunManager.PHASE_CARD_REWARD, combat_active=False,
        )
        curr = RunRewardSnapshot(
            total_floor=3, hp_ratio=1.0, max_hp=80,
            phase=RunManager.PHASE_MAP_CHOICE, combat_active=False,
        )
        reward = compute_navigator_shaping(prev, curr, config, draft_delta=0.25)
        assert reward == pytest.approx(compute_draft_value_shaping(0.25, config))

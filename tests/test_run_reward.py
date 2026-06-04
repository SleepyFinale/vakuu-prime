"""Tests for run-level reward shaping."""

import pytest

from sts2_env.gym_env.run_reward import (
    RunRewardConfig,
    RunRewardSnapshot,
    compute_run_shaping,
)
from sts2_env.run.run_manager import RunManager


class TestComputeRunShaping:
    def test_floor_bonus(self):
        config = RunRewardConfig(floor_bonus=0.05)
        prev = RunRewardSnapshot(
            total_floor=3, hp_ratio=1.0, phase=RunManager.PHASE_MAP_CHOICE,
            combat_active=False,
        )
        curr = RunRewardSnapshot(
            total_floor=5, hp_ratio=1.0, phase=RunManager.PHASE_MAP_CHOICE,
            combat_active=False,
        )
        assert compute_run_shaping(prev, curr, config) == pytest.approx(0.1)

    def test_combat_clear_and_hp_penalty(self):
        config = RunRewardConfig(
            combat_clear_bonus=0.1,
            hp_loss_penalty_scale=0.2,
            max_hp_loss_penalty=0.2,
        )
        prev = RunRewardSnapshot(
            total_floor=4, hp_ratio=0.8, phase=RunManager.PHASE_COMBAT,
            combat_active=True,
        )
        curr = RunRewardSnapshot(
            total_floor=4, hp_ratio=0.5, phase=RunManager.PHASE_CARD_REWARD,
            combat_active=False, last_combat_won=True,
        )
        reward = compute_run_shaping(prev, curr, config)
        assert reward == pytest.approx(0.1 - 0.06)

    def test_no_shaping_without_signals(self):
        config = RunRewardConfig()
        snap = RunRewardSnapshot(
            total_floor=2, hp_ratio=0.9, phase=RunManager.PHASE_MAP_CHOICE,
            combat_active=False,
        )
        assert compute_run_shaping(snap, snap, config) == 0.0


class TestActCurriculum:
    def test_max_acts_ends_run_after_first_act(self):
        mgr = RunManager(seed=99, character_id="Ironclad", max_acts=1)
        mgr._run_state.current_act_index = 0
        assert len(mgr.run_state.acts) > 1

        mgr._transition_next_act()

        assert mgr.is_over
        assert mgr.player_won
        assert mgr.phase == RunManager.PHASE_RUN_OVER

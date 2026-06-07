"""Tests for run-level reward shaping."""

import pytest

from sts2_env.gym_env.reward_shaping import HpShapingConfig, compute_hp_loss_penalty
from sts2_env.gym_env.run_reward import (
    REWARD_DEATH,
    REWARD_WIN,
    RunRewardConfig,
    RunRewardSnapshot,
    compute_run_shaping,
    compute_run_terminal_reward,
)
from sts2_env.run.run_manager import RunManager


class TestComputeRunShaping:
    def test_floor_bonus(self):
        config = RunRewardConfig(floor_bonus=0.01)
        prev = RunRewardSnapshot(
            total_floor=3, hp_ratio=1.0, max_hp=80,
            phase=RunManager.PHASE_MAP_CHOICE, combat_active=False,
        )
        curr = RunRewardSnapshot(
            total_floor=5, hp_ratio=1.0, max_hp=80,
            phase=RunManager.PHASE_MAP_CHOICE, combat_active=False,
        )
        assert compute_run_shaping(prev, curr, config) == pytest.approx(0.02)

    def test_combat_clear_and_hp_penalty(self):
        config = RunRewardConfig(
            combat_clear_bonus=0.1,
            hp=HpShapingConfig(penalty_scale=0.2, max_penalty=0.2, steepness=3.0),
        )
        max_hp = 100
        prev = RunRewardSnapshot(
            total_floor=4, hp_ratio=0.8, max_hp=max_hp,
            phase=RunManager.PHASE_COMBAT, combat_active=True,
        )
        curr = RunRewardSnapshot(
            total_floor=4, hp_ratio=0.5, max_hp=max_hp,
            phase=RunManager.PHASE_CARD_REWARD,
            combat_active=False, last_combat_won=True,
        )
        hp_lost = 30
        expected_penalty = compute_hp_loss_penalty(
            hp_lost, max_hp, prev.hp_ratio, config.hp,
        )
        reward = compute_run_shaping(prev, curr, config, combat_gross_hp_lost=30)
        assert reward == pytest.approx(0.1 - expected_penalty)

    def test_flawless_combat_clear_adds_bonus(self):
        config = RunRewardConfig(
            combat_clear_bonus=0.1,
            flawless_combat_bonus=0.05,
        )
        prev = RunRewardSnapshot(
            total_floor=4, hp_ratio=0.8, max_hp=100,
            phase=RunManager.PHASE_COMBAT, combat_active=True,
        )
        curr = RunRewardSnapshot(
            total_floor=4, hp_ratio=0.8, max_hp=100,
            phase=RunManager.PHASE_CARD_REWARD,
            combat_active=False, last_combat_won=True,
        )
        reward = compute_run_shaping(prev, curr, config, combat_gross_hp_lost=0)
        assert reward == pytest.approx(0.15)

    def test_damaged_combat_clear_has_no_flawless_bonus(self):
        config = RunRewardConfig(
            combat_clear_bonus=0.1,
            flawless_combat_bonus=0.05,
        )
        max_hp = 100
        prev = RunRewardSnapshot(
            total_floor=4, hp_ratio=0.8, max_hp=max_hp,
            phase=RunManager.PHASE_COMBAT, combat_active=True,
        )
        curr = RunRewardSnapshot(
            total_floor=4, hp_ratio=0.5, max_hp=max_hp,
            phase=RunManager.PHASE_CARD_REWARD,
            combat_active=False, last_combat_won=True,
        )
        reward = compute_run_shaping(prev, curr, config, combat_gross_hp_lost=30)
        hp_lost = 30
        expected_penalty = compute_hp_loss_penalty(
            hp_lost, max_hp, prev.hp_ratio, config.hp,
        )
        assert reward == pytest.approx(0.1 - expected_penalty)

    def test_no_shaping_without_signals(self):
        config = RunRewardConfig()
        snap = RunRewardSnapshot(
            total_floor=2, hp_ratio=0.9, max_hp=80,
            phase=RunManager.PHASE_MAP_CHOICE, combat_active=False,
        )
        assert compute_run_shaping(snap, snap, config) == 0.0


class TestRunTerminalReward:
    def test_win_hp_bonus_on_terminal_win(self):
        config = RunRewardConfig(win_hp_bonus_scale=0.15)
        assert compute_run_terminal_reward(
            player_won=True,
            hp_ratio=1.0,
            config=config,
            shaping_enabled=True,
        ) == pytest.approx(1.15)
        assert compute_run_terminal_reward(
            player_won=True,
            hp_ratio=0.5,
            config=config,
            shaping_enabled=True,
        ) == pytest.approx(1.075)

    def test_terminal_sparse_when_shaping_disabled(self):
        config = RunRewardConfig(win_hp_bonus_scale=0.15)
        assert compute_run_terminal_reward(
            player_won=True,
            hp_ratio=1.0,
            config=config,
            shaping_enabled=False,
        ) == pytest.approx(REWARD_WIN)
        assert compute_run_terminal_reward(
            player_won=False,
            hp_ratio=0.5,
            config=config,
            shaping_enabled=False,
        ) == pytest.approx(REWARD_DEATH)

    def test_incentive_invariant(self):
        """Max-progress loss must stay negative; minimal win must beat max loss."""
        config = RunRewardConfig()
        max_floors = 50
        max_combats = 35
        max_kills = 70

        progress = (
            max_floors * config.floor_bonus
            + max_combats * config.combat_clear_bonus
            + max_combats * config.flawless_combat_bonus
            + max_kills * config.micro.kill_scale
        )
        max_loss_total = progress + compute_run_terminal_reward(
            player_won=False,
            hp_ratio=0.0,
            config=config,
            shaping_enabled=True,
        )
        assert max_loss_total < 0.0

        min_win_progress = (
            1 * config.floor_bonus
            + 1 * config.combat_clear_bonus
            + 1 * config.flawless_combat_bonus
            + 2 * config.micro.kill_scale
        )
        min_win_total = min_win_progress + compute_run_terminal_reward(
            player_won=True,
            hp_ratio=1.0,
            config=config,
            shaping_enabled=True,
        )
        assert min_win_total > max_loss_total


class TestActCurriculum:
    def test_max_acts_ends_run_after_first_act(self):
        mgr = RunManager(seed=99, character_id="Ironclad", max_acts=1)
        mgr._run_state.current_act_index = 0
        assert len(mgr.run_state.acts) > 1

        mgr._transition_next_act()

        assert mgr.is_over
        assert mgr.player_won
        assert mgr.phase == RunManager.PHASE_RUN_OVER

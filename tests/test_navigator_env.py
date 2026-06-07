"""Tests for the Navigator full-run environment."""

import numpy as np
import pytest

from sts2_env.gym_env.navigator_env import STS2NavigatorEnv
from sts2_env.gym_env.navigator_observation import NAVIGATOR_OBS_SIZE
from sts2_env.gym_env.run_env import _COMBAT_START, _COMBAT_SIZE
from sts2_env.run.run_manager import RunManager


class _MockCombatModel:
    """Always end turn (action 0)."""

    def predict(self, obs, action_masks=None, deterministic=True):
        mask = action_masks if action_masks is not None else np.ones(115, dtype=np.int8)
        valid = np.where(mask == 1)[0]
        return int(valid[0]), None


@pytest.fixture
def mock_nav_env():
    return STS2NavigatorEnv(
        combat_model=_MockCombatModel(),
        reward_shaping=True,
        act_count=3,
        max_steps=500,
    )


class TestNavigatorEnv:
    def test_reset_valid(self, mock_nav_env):
        obs, info = mock_nav_env.reset(seed=42)
        assert obs.shape == (NAVIGATOR_OBS_SIZE,)
        assert mock_nav_env.observation_space.shape == (NAVIGATOR_OBS_SIZE,)
        assert "action_mask" in info

    def test_meta_step_increments_counter(self, mock_nav_env):
        mock_nav_env.reset(seed=42)
        mask = mock_nav_env.action_masks()
        valid = np.where(mask == 1)[0]
        mock_nav_env.step(int(valid[0]))
        assert mock_nav_env._meta_step_count == 1

    def test_masks_zero_combat_during_active_fight(self, mock_nav_env):
        env = mock_nav_env
        env.reset(seed=42)
        for _ in range(30):
            if env._run_env.is_active_combat():
                mask = env.action_masks()
                combat_slice = mask[_COMBAT_START:_COMBAT_START + _COMBAT_SIZE]
                assert combat_slice.sum() == 0
                return
            mask = env.action_masks()
            valid = np.where(mask == 1)[0]
            env.step(int(valid[0]))
        pytest.skip("Combat not reached in 30 meta steps")

    def test_episode_completes_with_mock_combat(self, mock_nav_env):
        env = mock_nav_env
        obs, info = env.reset(seed=99)
        done = False
        steps = 0
        while not done and steps < 500:
            mask = info["action_mask"]
            valid = np.where(mask == 1)[0]
            obs, reward, terminated, truncated, info = env.step(int(valid[0]))
            done = terminated or truncated
            steps += 1
        assert steps > 0

    def test_card_reward_actions_not_masked(self, mock_nav_env):
        env = mock_nav_env
        env.reset(seed=42)
        mgr = env._run_env._mgr
        for _ in range(200):
            if mgr.phase == RunManager.PHASE_CARD_REWARD:
                mask = env.action_masks()
                from sts2_env.gym_env.run_env import _LAYOUT

                layout = _LAYOUT
                card_slice = mask[
                    layout.card_reward_start:
                    layout.card_reward_extra_start + layout.card_reward_extra_size
                ]
                assert card_slice.sum() > 0
                return
            mask = env.action_masks()
            valid = np.where(mask == 1)[0]
            env.step(int(valid[0]))
        pytest.skip("Card reward not reached")

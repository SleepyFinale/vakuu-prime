"""Tests for the hierarchical full-run environment."""

import numpy as np
import pytest

from sts2_env.gym_env.hierarchical_run_env import STS2HierarchicalRunEnv
from sts2_env.gym_env.run_env import STS2RunEnv, _COMBAT_START, _COMBAT_SIZE
from sts2_env.run.run_manager import RunManager


class _MockCombatModel:
    """Always end turn (action 0)."""

    def predict(self, obs, action_masks=None, deterministic=True):
        mask = action_masks if action_masks is not None else np.ones(115, dtype=np.int8)
        valid = np.where(mask == 1)[0]
        return int(valid[0]), None


@pytest.fixture
def mock_hier_env():
    return STS2HierarchicalRunEnv(
        delegate_combat=True,
        combat_model=_MockCombatModel(),
        reward_shaping=True,
        act_count=3,
        max_steps=500,
    )


class TestHierarchicalRunEnv:
    def test_reset_valid(self, mock_hier_env):
        obs, info = mock_hier_env.reset(seed=42)
        assert obs.shape == mock_hier_env.observation_space.shape
        assert "action_mask" in info

    def test_meta_step_increments_counter(self, mock_hier_env):
        mock_hier_env.reset(seed=42)
        mask = mock_hier_env.action_masks()
        valid = np.where(mask == 1)[0]
        mock_hier_env.step(int(valid[0]))
        assert mock_hier_env._meta_step_count == 1

    def test_masks_zero_combat_during_active_fight(self, mock_hier_env):
        env = mock_hier_env
        env.reset(seed=42)
        mgr = env._run_env._mgr
        # Advance into combat if possible
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

    def test_episode_completes_with_mock_combat(self, mock_hier_env):
        env = mock_hier_env
        obs, info = env.reset(seed=7)
        done = False
        steps = 0
        while not done and steps < 500:
            mask = info["action_mask"]
            valid = np.where(mask == 1)[0]
            obs, reward, terminated, truncated, info = env.step(int(valid[0]))
            done = terminated or truncated
            steps += 1
        assert done

    def test_flat_run_env_accepts_new_kwargs(self):
        env = STS2RunEnv(reward_shaping=True, act_count=1)
        obs, info = env.reset(seed=0)
        assert obs.shape[0] > 0
        mask = info["action_mask"]
        valid = np.where(mask == 1)[0]
        obs, reward, term, trunc, info = env.step(int(valid[0]))
        assert isinstance(reward, float)

    def test_run_state_property(self, mock_hier_env):
        mock_hier_env.reset(seed=1)
        assert mock_hier_env.run_state is not None

    def test_noncombat_heuristic_auto_resolves_card_reward(self):
        from sts2_env.gym_env.hierarchical_run_env import STS2HierarchicalRunEnv
        from sts2_env.run.reward_objects import CardReward

        env = STS2HierarchicalRunEnv(
            delegate_combat=True,
            combat_model=_MockCombatModel(),
            use_noncombat_heuristic=True,
            max_steps=500,
        )
        env.reset(seed=42)
        mgr = env._run_env._mgr
        mgr._phase = RunManager.PHASE_CARD_REWARD
        reward = CardReward(mgr.run_state.player.player_id)
        reward.populate(mgr.run_state, None)
        mgr._current_reward = reward
        mgr._offered_cards = list(reward.cards)

        meta_steps_before = env._meta_step_count
        shaping_reward = env._auto_resolve_noncombat()
        assert shaping_reward >= 0.0
        assert env._meta_step_count == meta_steps_before
        assert mgr.phase != RunManager.PHASE_CARD_REWARD

"""Tests for multi-act STS2CombatEnv."""

import pytest

from sts2_env.encounters.pools import build_encounter_pool
from sts2_env.gym_env.combat_env import STS2CombatEnv


def test_combat_env_reset_mixed_acts():
    env = STS2CombatEnv(encounter_acts=(0, 1, 2))
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert len(env.encounter_pool) == len(build_encounter_pool((0, 1, 2), include_boss=False))


def test_combat_env_episode_completes():
    env = STS2CombatEnv(encounter_acts=(0, 1))
    obs, info = env.reset(seed=7)
    done = False
    steps = 0
    while not done and steps < 500:
        mask = info["action_mask"]
        import numpy as np
        valid = np.where(mask == 1)[0]
        obs, reward, terminated, truncated, info = env.step(int(valid[0]))
        done = terminated or truncated
        steps += 1
    assert done

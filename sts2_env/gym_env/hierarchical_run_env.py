"""Hierarchical full-run environment with frozen combat delegation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np

from sts2_env.gym_env.observation import OBS_SIZE as COMBAT_OBS_SIZE
from sts2_env.gym_env.run_env import (
    DEFAULT_MAX_STEPS,
    REWARD_DEATH,
    REWARD_WIN,
    STS2RunEnv,
    _LAYOUT,
)
from sts2_env.gym_env.run_reward import (
    RunRewardConfig,
    compute_run_shaping,
    snapshot_from_manager,
)
from sts2_env.run.run_manager import RunManager

logger = logging.getLogger(__name__)

INNER_MAX_STEPS = DEFAULT_MAX_STEPS * 20


def load_combat_model(model_path: str | Path):
    """Load a trained MaskablePPO combat policy."""
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sts2_env.gym_env.combat_env import STS2CombatEnv

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Combat model not found: {path}")

    def mask_fn(env):
        return env.action_masks()

    dummy_env = ActionMasker(STS2CombatEnv(), mask_fn)
    return MaskablePPO.load(str(path), env=dummy_env)


class STS2HierarchicalRunEnv(gymnasium.Env):
    """Full-run env that auto-plays combat with a frozen combat PPO.

    One ``step()`` = one meta decision (map, rewards, shop, etc.) plus
    any combat turns resolved internally by the combat sub-policy.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        combat_model_path: str | Path | None = None,
        delegate_combat: bool = True,
        character_id: str = "Ironclad",
        ascension_level: int = 0,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_combat_turns: int = 200,
        reward_shaping: bool = True,
        act_count: int = 3,
        reward_config: RunRewardConfig | None = None,
        render_mode: str | None = None,
        combat_model: Any | None = None,
    ):
        super().__init__()
        self._run_env = STS2RunEnv(
            character_id=character_id,
            ascension_level=ascension_level,
            max_steps=INNER_MAX_STEPS,
            max_combat_turns=max_combat_turns,
            reward_shaping=reward_shaping,
            act_count=act_count,
            reward_config=reward_config,
            render_mode=render_mode,
        )
        self.observation_space = self._run_env.observation_space
        self.action_space = self._run_env.action_space
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.delegate_combat = delegate_combat
        self.reward_shaping = reward_shaping
        self._reward_config = self._run_env._reward_config

        self._combat_model = combat_model
        self._combat_model_path = combat_model_path
        if delegate_combat and self._combat_model is None:
            if combat_model_path is None:
                raise ValueError(
                    "combat_model_path is required when delegate_combat=True"
                )
            self._combat_model_path = combat_model_path

        self._meta_step_count = 0
        self._layout = _LAYOUT

    @property
    def run_state(self):
        return self._run_env.run_state

    @property
    def run_env(self) -> STS2RunEnv:
        return self._run_env

    def _ensure_combat_model(self) -> None:
        if not self.delegate_combat:
            return
        if self._combat_model is None:
            self._combat_model = load_combat_model(self._combat_model_path)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        obs, info = self._run_env.reset(seed=seed)
        self._meta_step_count = 0
        return obs, info

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._run_env._mgr is not None
        self._meta_step_count += 1

        mgr = self._run_env._mgr
        prev_snapshot = snapshot_from_manager(mgr)
        reward = 0.0

        self._run_env._dispatch_action(action)
        if self.reward_shaping:
            curr_snapshot = snapshot_from_manager(mgr)
            reward += compute_run_shaping(
                prev_snapshot, curr_snapshot, self._reward_config,
            )
            self._run_env._prev_reward_snapshot = curr_snapshot

        if self.delegate_combat:
            reward += self._auto_play_combat()

        terminated = mgr.is_over
        truncated = self._meta_step_count >= self.max_steps and not terminated

        if terminated:
            reward += REWARD_WIN if mgr.player_won else REWARD_DEATH
        elif truncated:
            reward += REWARD_DEATH

        obs = self._run_env._encode_obs()
        info = self._run_env._build_info()
        info["meta_step"] = self._meta_step_count
        return obs, float(reward), terminated, truncated, info

    def _auto_play_combat(self) -> float:
        """Resolve combat turns until combat ends or the run is over."""
        self._ensure_combat_model()
        layout = self._layout
        total_shaping = 0.0
        max_inner = INNER_MAX_STEPS

        for _ in range(max_inner):
            if self._run_env._mgr is None or self._run_env._mgr.is_over:
                break
            if not self._run_env.is_active_combat():
                break

            prev_snapshot = snapshot_from_manager(self._run_env._mgr)

            if self._run_env.needs_player_select():
                action = layout.player_select_start
            else:
                obs = self._run_env._encode_obs()
                full_mask = self._run_env.action_masks()
                combat_obs = obs[:COMBAT_OBS_SIZE]
                combat_mask = full_mask[
                    layout.combat_start: layout.combat_start + layout.combat_size
                ]
                local_action, _ = self._combat_model.predict(
                    combat_obs,
                    action_masks=combat_mask,
                    deterministic=True,
                )
                action = layout.combat_start + int(local_action)

            self._run_env._dispatch_action(action)

            if self.reward_shaping:
                curr_snapshot = snapshot_from_manager(self._run_env._mgr)
                total_shaping += compute_run_shaping(
                    prev_snapshot, curr_snapshot, self._reward_config,
                )
                self._run_env._prev_reward_snapshot = curr_snapshot

        return total_shaping

    def action_masks(self) -> np.ndarray:
        mask = self._run_env.action_masks()
        if self._run_env.is_active_combat():
            layout = self._layout
            mask[layout.combat_start: layout.combat_start + layout.combat_size] = 0
            if self._run_env.needs_player_select():
                for i in range(layout.player_select_size):
                    mask[layout.player_select_start + i] = 0
        if mask.sum() == 0:
            mask[0] = 1
        return mask

    def render(self) -> str | None:
        return self._run_env.render()

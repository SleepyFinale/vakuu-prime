"""Navigator environment: strategic run control with frozen combat delegation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from sts2_env.gym_env.combat_value import CombatValueConfig, draft_value_for_pick
from sts2_env.gym_env.hierarchical_run_env import (
    INNER_MAX_STEPS,
    load_combat_model,
    load_combat_models,
    load_combat_models_by_character,
)
from sts2_env.gym_env.navigator_observation import (
    NAVIGATOR_OBS_SIZE,
    encode_navigator_observation,
)
from sts2_env.gym_env.observation import OBS_SIZE as COMBAT_OBS_SIZE
from sts2_env.gym_env.run_env import (
    DEFAULT_MAX_STEPS,
    STS2RunEnv,
    TOTAL_ACTIONS,
    _LAYOUT,
)
from sts2_env.gym_env.run_reward import (
    NavigatorRewardConfig,
    REWARD_DEATH,
    RunRewardConfig,
    compute_navigator_shaping,
    compute_run_terminal_reward,
    snapshot_from_manager,
)
from sts2_env.run.run_manager import RunManager

logger = logging.getLogger(__name__)


def _card_pick_index_from_action(action: int, layout=_LAYOUT) -> int | None:
    """Map a global card-reward action to pick index, or None for skip."""
    if action == layout.card_reward_start + 3:
        return None
    if layout.card_reward_start <= action < layout.card_reward_start + layout.card_reward_size:
        return action - layout.card_reward_start
    if layout.card_reward_extra_start <= action < layout.card_reward_extra_start + layout.card_reward_extra_size:
        return 3 + (action - layout.card_reward_extra_start)
    return None


class STS2NavigatorEnv(gymnasium.Env):
    """Strategic run env: Navigator owns all non-combat phases; combat is delegated."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        combat_model_path: str | Path | None = None,
        combat_models: dict[int, str | Path] | None = None,
        combat_models_by_character: dict[str, str | Path] | None = None,
        character_id: str = "Ironclad",
        character_ids: tuple[str, ...] | None = None,
        ascension_level: int = 0,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_combat_turns: int = 200,
        reward_shaping: bool = True,
        act_count: int = 3,
        reward_config: RunRewardConfig | NavigatorRewardConfig | None = None,
        render_mode: str | None = None,
        act1_biome: str = "random",
        underdocks_unlocked: bool = True,
        underdocks_discovered: bool = True,
        combat_model: Any | None = None,
        combat_models_loaded: dict[int, Any] | None = None,
        combat_models_by_character_loaded: dict[str, Any] | None = None,
        combat_value_shaping: bool = False,
        combat_value_config: CombatValueConfig | None = None,
        deck_value_in_obs: bool = False,
    ):
        super().__init__()
        if reward_config is None and reward_shaping:
            reward_config = NavigatorRewardConfig()
        elif reward_config is not None and not isinstance(reward_config, NavigatorRewardConfig):
            reward_config = NavigatorRewardConfig(
                floor_bonus=reward_config.floor_bonus,
                combat_clear_bonus=reward_config.combat_clear_bonus,
                hp=reward_config.hp,
                micro=reward_config.micro,
            )

        self._run_env = STS2RunEnv(
            character_id=character_id,
            character_ids=character_ids,
            ascension_level=ascension_level,
            max_steps=INNER_MAX_STEPS,
            max_combat_turns=max_combat_turns,
            reward_shaping=reward_shaping,
            act_count=act_count,
            reward_config=reward_config,
            render_mode=render_mode,
            act1_biome=act1_biome,
            underdocks_unlocked=underdocks_unlocked,
            underdocks_discovered=underdocks_discovered,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=10.0,
            shape=(NAVIGATOR_OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        self._reward_config = reward_config or NavigatorRewardConfig()
        self.combat_value_shaping = combat_value_shaping
        self._combat_value_config = combat_value_config or CombatValueConfig()
        self.deck_value_in_obs = deck_value_in_obs

        self._combat_model = combat_model
        self._combat_models: dict[int, Any] = dict(combat_models_loaded or {})
        self._combat_models_by_character: dict[str, Any] = dict(
            combat_models_by_character_loaded or {},
        )
        self._combat_model_path = combat_model_path
        self._combat_models_paths = combat_models
        self._combat_models_by_character_paths = combat_models_by_character

        if combat_models_by_character and not self._combat_models_by_character:
            self._combat_models_by_character = load_combat_models_by_character(
                combat_models_by_character,
            )
        elif combat_models and not self._combat_models:
            self._combat_models = load_combat_models(
                combat_models,
                character_id=character_id,
            )
        elif self._combat_model is None and combat_model_path is not None:
            self._combat_model = load_combat_model(
                combat_model_path,
                character_id=character_id,
            )
        elif (
            self._combat_model is None
            and combat_model_path is None
            and not self._combat_models
            and not self._combat_models_by_character
        ):
            raise ValueError(
                "combat_model_path, combat_models, or combat_models_by_character required"
            )

        self._meta_step_count = 0
        self._layout = _LAYOUT
        self._cached_deck_value = 0.0

    @property
    def run_state(self):
        return self._run_env.run_state

    @property
    def run_env(self) -> STS2RunEnv:
        return self._run_env

    def _ensure_combat_models(self) -> None:
        if self._combat_models_by_character:
            return
        if self._combat_models:
            return
        if self._combat_model is not None:
            return
        if self._combat_models_by_character_paths:
            self._combat_models_by_character = load_combat_models_by_character(
                self._combat_models_by_character_paths,
            )
            return
        if self._combat_models_paths:
            self._combat_models = load_combat_models(
                self._combat_models_paths,
                character_id=self._run_env._character_id,
            )
            return
        if self._combat_model_path is not None:
            self._combat_model = load_combat_model(
                self._combat_model_path,
                character_id=self._run_env._character_id,
            )

    def _combat_policy_for_act(self, act_index: int) -> Any:
        self._ensure_combat_models()
        if self._combat_models:
            if act_index in self._combat_models:
                return self._combat_models[act_index]
            if 0 in self._combat_models:
                return self._combat_models[0]
            return next(iter(self._combat_models.values()))
        return self._combat_model

    def _combat_policy_for_character(self, character_id: str) -> Any:
        self._ensure_combat_models()
        if self._combat_models_by_character:
            if character_id in self._combat_models_by_character:
                return self._combat_models_by_character[character_id]
            if "Ironclad" in self._combat_models_by_character:
                return self._combat_models_by_character["Ironclad"]
            return next(iter(self._combat_models_by_character.values()))
        return self._combat_policy_for_act(0)

    def _combat_policy_for_run(self, mgr: RunManager) -> Any:
        character_id = mgr.run_state.player.character_id
        if self._combat_models_by_character or self._combat_models_by_character_paths:
            return self._combat_policy_for_character(character_id)
        return self._combat_policy_for_act(mgr.run_state.current_act_index)

    def _encode_obs(self) -> np.ndarray:
        mgr = self._run_env._mgr
        deck_value = self._cached_deck_value if self.deck_value_in_obs else 0.0
        return encode_navigator_observation(mgr, deck_value=deck_value)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        _, info = self._run_env.reset(seed=seed)
        self._meta_step_count = 0
        self._cached_deck_value = 0.0
        obs = self._encode_obs()
        info["action_mask"] = self.action_masks()
        return obs, info

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._run_env._mgr is not None
        self._meta_step_count += 1
        mgr = self._run_env._mgr

        draft_delta = 0.0
        if (
            self.combat_value_shaping
            and mgr.phase == RunManager.PHASE_CARD_REWARD
        ):
            pick_index = _card_pick_index_from_action(action, self._layout)
            if pick_index is not None or action == self._layout.card_reward_start + 3:
                self._ensure_combat_models()
                draft_delta = draft_value_for_pick(
                    mgr,
                    pick_index,
                    self._combat_policy_for_run(mgr),
                    config=self._combat_value_config,
                )

        prev_snapshot = snapshot_from_manager(mgr)
        prev_combat_hp = self._run_env._combat_hp_before_step()
        self._run_env._dispatch_action(action)
        reward = self._shaping_delta(
            prev_snapshot, prev_combat_hp=prev_combat_hp, draft_delta=draft_delta,
        )
        reward += self._auto_play_combat()

        terminated = mgr.is_over
        truncated = self._meta_step_count >= self.max_steps and not terminated
        if terminated:
            hp_ratio = snapshot_from_manager(mgr).hp_ratio
            reward += compute_run_terminal_reward(
                player_won=mgr.player_won,
                hp_ratio=hp_ratio,
                config=self._reward_config,
                shaping_enabled=self.reward_shaping,
            )
        elif truncated:
            reward += REWARD_DEATH

        obs = self._encode_obs()
        info = self._run_env._build_info()
        info["meta_step"] = self._meta_step_count
        if draft_delta != 0.0:
            info["draft_value_delta"] = draft_delta
        info["action_mask"] = self.action_masks()
        return obs, float(reward), terminated, truncated, info

    def _shaping_delta(
        self,
        prev_snapshot,
        *,
        draft_delta: float = 0.0,
        prev_combat_hp: int | None = None,
    ) -> float:
        if not self.reward_shaping:
            return 0.0
        assert self._run_env._mgr is not None
        self._run_env._accumulate_combat_hp_lost(prev_combat_hp)
        curr_snapshot = snapshot_from_manager(self._run_env._mgr)
        config = self._reward_config
        if isinstance(config, NavigatorRewardConfig):
            delta = compute_navigator_shaping(
                prev_snapshot,
                curr_snapshot,
                config,
                draft_delta=draft_delta,
                combat_gross_hp_lost=self._run_env._combat_gross_hp_lost,
            )
        else:
            from sts2_env.gym_env.run_reward import compute_run_shaping

            delta = compute_run_shaping(
                prev_snapshot,
                curr_snapshot,
                config,
                combat_gross_hp_lost=self._run_env._combat_gross_hp_lost,
            )
        self._run_env._reset_combat_shaping_on_phase_change(
            prev_snapshot, curr_snapshot,
        )
        self._run_env._prev_reward_snapshot = curr_snapshot
        return delta

    def _auto_play_combat(self) -> float:
        self._ensure_combat_models()
        layout = self._layout
        total_shaping = 0.0

        for _ in range(INNER_MAX_STEPS):
            mgr = self._run_env._mgr
            if mgr is None or mgr.is_over:
                break
            if not self._run_env.is_active_combat():
                break

            prev_snapshot = snapshot_from_manager(mgr)
            combat_policy = self._combat_policy_for_run(mgr)

            if self._run_env.needs_player_select():
                action = layout.player_select_start
            else:
                obs = self._run_env._encode_obs()
                full_mask = self._run_env.action_masks()
                combat_obs = obs[:COMBAT_OBS_SIZE]
                combat_mask = full_mask[
                    layout.combat_start: layout.combat_start + layout.combat_size
                ]
                local_action, _ = combat_policy.predict(
                    combat_obs,
                    action_masks=combat_mask,
                    deterministic=True,
                )
                action = layout.combat_start + int(local_action)

            prev_combat_hp = self._run_env._combat_hp_before_step()
            self._run_env._dispatch_action(action)
            total_shaping += self._shaping_delta(
                prev_snapshot, prev_combat_hp=prev_combat_hp,
            )

        return total_shaping

    def action_masks(self) -> np.ndarray:
        mask = self._run_env.action_masks()
        layout = self._layout
        mask[layout.combat_start: layout.combat_start + layout.combat_size] = 0
        if self._run_env.is_active_combat():
            if self._run_env.needs_player_select():
                for i in range(layout.player_select_size):
                    mask[layout.player_select_start + i] = 0
        if mask.sum() == 0:
            mask[0] = 1
        return mask

    def render(self) -> str | None:
        return self._run_env.render()

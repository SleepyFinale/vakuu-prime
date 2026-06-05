"""Hierarchical full-run environment with frozen combat delegation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import gymnasium
import numpy as np

from sts2_env.gym_env.noncombat_heuristics import (
    NoncombatHeuristicConfig,
    heuristic_global_action,
    should_auto_resolve_phase,
)
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


def parse_combat_models_spec(spec: str) -> dict[int, Path]:
    """Parse ``'0:path0,1:path1'`` into act-indexed model paths."""
    models: dict[int, Path] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        act_str, _, path_str = part.partition(":")
        if not path_str:
            raise ValueError(
                f"Invalid combat-models entry {part!r}; expected 'act_index:path'"
            )
        models[int(act_str.strip())] = Path(path_str.strip())
    if not models:
        raise ValueError(f"No combat models parsed from: {spec!r}")
    return models


def parse_combat_models_by_character(spec: str) -> dict[str, Path]:
    """Parse ``'Ironclad:path0,Silent:path1'`` into character-indexed model paths."""
    from sts2_env.characters.all import get_character

    models: dict[str, Path] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        char_str, _, path_str = part.partition(":")
        if not path_str:
            raise ValueError(
                f"Invalid combat-models-by-character entry {part!r}; "
                "expected 'CharacterId:path'"
            )
        char_id = get_character(char_str.strip()).character_id
        models[char_id] = Path(path_str.strip())
    if not models:
        raise ValueError(f"No combat models parsed from: {spec!r}")
    return models


def load_combat_model(
    model_path: str | Path,
    encounter_acts: tuple[int, ...] = (0,),
    character_id: str = "Ironclad",
):
    """Load a trained MaskablePPO combat policy."""
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sts2_env.gym_env.combat_env import STS2CombatEnv

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Combat model not found: {path}")

    def mask_fn(env):
        return env.action_masks()

    dummy_env = ActionMasker(
        STS2CombatEnv(
            encounter_acts=encounter_acts,
            character_id=character_id,
        ),
        mask_fn,
    )
    return MaskablePPO.load(str(path), env=dummy_env)


def load_combat_models(
    models: dict[int, str | Path],
    *,
    default_encounter_acts: tuple[int, ...] = (0,),
    character_id: str = "Ironclad",
) -> dict[int, Any]:
    """Load one combat policy per act index."""
    loaded: dict[int, Any] = {}
    for act_index, path in models.items():
        acts = (act_index,) if act_index in default_encounter_acts else default_encounter_acts
        loaded[act_index] = load_combat_model(
            path, encounter_acts=acts, character_id=character_id,
        )
    return loaded


def load_combat_models_by_character(
    models: dict[str, str | Path],
    *,
    default_encounter_acts: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    """Load one combat policy per character id."""
    loaded: dict[str, Any] = {}
    for char_id, path in models.items():
        loaded[char_id] = load_combat_model(
            path,
            encounter_acts=default_encounter_acts,
            character_id=char_id,
        )
    return loaded


class STS2HierarchicalRunEnv(gymnasium.Env):
    """Full-run env that auto-plays combat with a frozen combat PPO.

    One ``step()`` = one meta decision (map, shop, events, etc.) plus
    automatic resolution of card rewards / rest / boss relic (optional)
    and combat turns via the combat sub-policy.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        combat_model_path: str | Path | None = None,
        combat_models: dict[int, str | Path] | None = None,
        delegate_combat: bool = True,
        use_noncombat_heuristic: bool = True,
        noncombat_heuristic_config: NoncombatHeuristicConfig | None = None,
        character_id: str = "Ironclad",
        character_ids: tuple[str, ...] | None = None,
        ascension_level: int = 0,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_combat_turns: int = 200,
        reward_shaping: bool = True,
        act_count: int = 3,
        reward_config: RunRewardConfig | None = None,
        render_mode: str | None = None,
        act1_biome: str = "random",
        underdocks_unlocked: bool = True,
        underdocks_discovered: bool = True,
        combat_model: Any | None = None,
        combat_models_loaded: dict[int, Any] | None = None,
        combat_models_by_character: dict[str, str | Path] | None = None,
        combat_models_by_character_loaded: dict[str, Any] | None = None,
        card_value_model_path: str | Path | None = None,
        card_value_model: Any | None = None,
        card_reward_observer: Callable[[RunManager], None] | None = None,
    ):
        super().__init__()
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
        self.observation_space = self._run_env.observation_space
        self.action_space = self._run_env.action_space
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.delegate_combat = delegate_combat
        self.use_noncombat_heuristic = use_noncombat_heuristic
        self._heuristic_config = noncombat_heuristic_config or NoncombatHeuristicConfig()
        self._card_reward_observer = card_reward_observer
        self._card_value_model = card_value_model
        self._card_value_model_path = card_value_model_path
        self._card_value_config = None
        if card_value_model_path is not None or card_value_model is not None:
            self._configure_learned_card_picker(card_value_model, card_value_model_path)
        self.reward_shaping = reward_shaping
        self._reward_config = self._run_env._reward_config

        self._combat_model = combat_model
        self._combat_models: dict[int, Any] = dict(combat_models_loaded or {})
        self._combat_models_by_character: dict[str, Any] = dict(
            combat_models_by_character_loaded or {},
        )
        self._combat_model_path = combat_model_path
        self._combat_models_paths = combat_models
        self._combat_models_by_character_paths = combat_models_by_character

        if delegate_combat:
            if combat_models_by_character and not self._combat_models_by_character:
                self._combat_models_by_character = load_combat_models_by_character(
                    combat_models_by_character,
                )
            elif combat_models and not self._combat_models:
                self._combat_models = load_combat_models(
                    combat_models,
                    character_id=character_id,
                )
            elif (
                self._combat_model is None
                and combat_model_path is None
                and not self._combat_models
                and not self._combat_models_by_character
            ):
                raise ValueError(
                    "combat_model_path, combat_models, or combat_models_by_character "
                    "required when delegate_combat=True"
                )
            if combat_model_path is not None:
                self._combat_model_path = combat_model_path

        self._meta_step_count = 0
        self._layout = _LAYOUT

    @property
    def run_state(self):
        return self._run_env.run_state

    @property
    def run_env(self) -> STS2RunEnv:
        return self._run_env

    def _ensure_combat_models(self) -> None:
        if not self.delegate_combat:
            return
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

    def _configure_learned_card_picker(
        self,
        card_value_model: Any | None,
        card_value_model_path: str | Path | None,
    ) -> None:
        from sts2_env.gym_env.card_value import load_card_value_model

        if card_value_model is not None:
            self._card_value_model = card_value_model
        elif card_value_model_path is not None:
            self._card_value_model, self._card_value_config = load_card_value_model(
                card_value_model_path,
            )
        self._heuristic_config.card_reward_mode = "learned"
        self._heuristic_config.card_value_model = self._card_value_model
        self._heuristic_config.card_value_config = self._card_value_config

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
        reward = self._apply_action_with_automation(action)

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

    def _apply_action_with_automation(self, action: int) -> float:
        """Dispatch meta action, then auto-resolve heuristics and combat."""
        mgr = self._run_env._mgr
        assert mgr is not None

        prev_snapshot = snapshot_from_manager(mgr)
        self._run_env._dispatch_action(action)
        reward = self._shaping_delta(prev_snapshot)

        if self.use_noncombat_heuristic:
            reward += self._auto_resolve_noncombat()
        if self.delegate_combat:
            reward += self._auto_play_combat()

        return reward

    def _shaping_delta(self, prev_snapshot) -> float:
        if not self.reward_shaping:
            return 0.0
        assert self._run_env._mgr is not None
        curr_snapshot = snapshot_from_manager(self._run_env._mgr)
        delta = compute_run_shaping(
            prev_snapshot, curr_snapshot, self._reward_config,
        )
        self._run_env._prev_reward_snapshot = curr_snapshot
        return delta

    def _auto_resolve_noncombat(self) -> float:
        """Apply heuristic picks for card reward, boss relic, and rest."""
        total_shaping = 0.0
        max_inner = INNER_MAX_STEPS

        for _ in range(max_inner):
            mgr = self._run_env._mgr
            if mgr is None or mgr.is_over:
                break
            if not should_auto_resolve_phase(mgr, self._heuristic_config):
                break

            if (
                self._card_reward_observer is not None
                and mgr.phase == RunManager.PHASE_CARD_REWARD
                and mgr._offered_cards
            ):
                self._card_reward_observer(mgr)

            action = heuristic_global_action(mgr, self._heuristic_config, self._layout)
            if action is None:
                break

            prev_snapshot = snapshot_from_manager(mgr)
            self._run_env._dispatch_action(action)
            total_shaping += self._shaping_delta(prev_snapshot)

        return total_shaping

    def _auto_play_combat(self) -> float:
        """Resolve combat turns until combat ends or the run is over."""
        self._ensure_combat_models()
        layout = self._layout
        total_shaping = 0.0
        max_inner = INNER_MAX_STEPS

        for _ in range(max_inner):
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

            self._run_env._dispatch_action(action)
            total_shaping += self._shaping_delta(prev_snapshot)

        return total_shaping

    def action_masks(self) -> np.ndarray:
        mask = self._run_env.action_masks()
        mgr = self._run_env._mgr
        if mgr is not None and should_auto_resolve_phase(mgr, self._heuristic_config):
            layout = self._layout
            if mgr.phase == RunManager.PHASE_CARD_REWARD:
                mask[layout.card_reward_start: layout.card_reward_extra_start + layout.card_reward_extra_size] = 0
                mask[layout.card_reward_reroll] = 0
            elif mgr.phase == RunManager.PHASE_BOSS_RELIC:
                mask[layout.boss_relic_start: layout.boss_relic_start + layout.boss_relic_size] = 0
            elif mgr.phase == RunManager.PHASE_REST_SITE:
                mask[layout.rest_start: layout.rest_start + layout.rest_size] = 0
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

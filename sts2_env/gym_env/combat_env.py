"""STS2 Combat Gymnasium Environment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium
import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from sts2_env.training.combat_curriculum import (
        CombatCurriculumStage,
        EpisodeInitSample,
    )

from sts2_env.cards.base import reset_instance_counter
from sts2_env.characters.all import (
    create_starting_deck,
    get_character,
    resolve_character_for_episode,
)
from sts2_env.core.combat import CombatState
from sts2_env.core.constants import ACTION_END_TURN, ACTION_SPACE_SIZE
from sts2_env.encounters.pools import (
    build_encounter_pool,
    build_mixed_act1_encounter_pool,
)
from sts2_env.encounters.registry import EncounterSetup
from sts2_env.core.rng import INT_MAX_EXCLUSIVE, Rng
from sts2_env.gym_env.action_space import (
    action_to_card_and_target,
    action_to_potion_and_target,
    get_action_mask,
    is_potion_action,
)
from sts2_env.gym_env.observation import OBS_SIZE, encode_observation
from sts2_env.gym_env.reward import compute_reward
from sts2_env.gym_env.reward_shaping import CombatEventCursor, CombatRewardConfig

logger = logging.getLogger(__name__)


class STS2CombatEnv(gymnasium.Env):
    """Gymnasium environment for a single STS2 combat encounter.

    Observation: flat float32 vector encoding player state, hand, piles, enemies,
    and character-specific mechanics (orbs, stars, Osty).
    Action: fixed discrete combat action space including cards, end turn, and potions.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        encounter_pool: list[EncounterSetup] | None = None,
        encounter_acts: tuple[int, ...] = (0,),
        act1_biome: str = "random",
        character_id: str = "Ironclad",
        character_ids: tuple[str, ...] | None = None,
        player_hp: int | None = None,
        player_max_hp: int | None = None,
        max_turns: int = 200,
        render_mode: str | None = None,
        reward_shaping: bool = True,
        reward_config: CombatRewardConfig | None = None,
        curriculum_stage: CombatCurriculumStage | None = None,
        curriculum_sequence: tuple[CombatCurriculumStage, ...] | None = None,
        curriculum_state_path: str | Path | None = None,
        hard_start_fraction: float | None = None,
        ascension_level: int = 0,
    ):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        if encounter_pool is not None:
            self.encounter_pool = encounter_pool
        elif act1_biome == "random" and 0 in encounter_acts:
            self.encounter_pool = build_mixed_act1_encounter_pool(
                encounter_acts, include_boss=False,
            )
        else:
            self.encounter_pool = build_encounter_pool(
                encounter_acts,
                include_boss=False,
                act1_biome=act1_biome,
            )
        self.encounter_acts = encounter_acts
        self.act1_biome = act1_biome
        self.character_id = character_id
        self.character_ids = character_ids
        self._fixed_player_hp = player_hp
        self._fixed_player_max_hp = player_max_hp
        self.max_turns = max_turns
        self.render_mode = render_mode
        self.reward_shaping = reward_shaping
        self._reward_config = reward_config or CombatRewardConfig()

        self.curriculum_stage = curriculum_stage
        self.curriculum_sequence = curriculum_sequence
        self.curriculum_state_path = (
            Path(curriculum_state_path) if curriculum_state_path is not None else None
        )
        self.hard_start_fraction = hard_start_fraction
        self._ascension_level = ascension_level

        self.combat: CombatState | None = None
        self._last_character_id: str | None = None
        self._last_episode_init: EpisodeInitSample | None = None
        self._event_cursor = CombatEventCursor()
        self._combat_gross_hp_lost = 0

    def _resolve_character_pool(self) -> tuple[str, ...]:
        if self.character_ids is not None:
            return self.character_ids
        return (self.character_id,)

    def _resolve_curriculum_stage(self) -> CombatCurriculumStage | None:
        if self.curriculum_stage is not None:
            return self.curriculum_stage
        if self.curriculum_state_path is None or self.curriculum_sequence is None:
            return None
        from sts2_env.training.curriculum_env import read_curriculum_stage_index

        index = read_curriculum_stage_index(self.curriculum_state_path)
        if index < 0 or index >= len(self.curriculum_sequence):
            index = 0
        return self.curriculum_sequence[index]

    def _apply_episode_init(self, episode_init: EpisodeInitSample, rng_seed: int) -> None:
        char_cfg = get_character(episode_init.character_id)
        self._last_character_id = episode_init.character_id
        self._last_episode_init = episode_init
        self.combat = CombatState(
            player_hp=episode_init.player_hp,
            player_max_hp=episode_init.player_max_hp,
            deck=list(episode_init.deck),
            rng_seed=rng_seed,
            character_id=episode_init.character_id,
            relics=[char_cfg.starting_relic],
            ascension_level=self._ascension_level,
        )
        rng = Rng(rng_seed)
        episode_init.encounter_setup(self.combat, rng)
        self.combat.start_combat()

    def _build_episode_info(self) -> dict:
        from sts2_env.training.combat_curriculum import encounter_setup_name

        info: dict = {
            "action_mask": get_action_mask(self.combat),
            "character_id": self._last_character_id,
        }
        if self._last_episode_init is not None:
            info.update(
                is_hard_start=self._last_episode_init.is_hard_start,
                deck_template=self._last_episode_init.deck_template,
                encounter_id=encounter_setup_name(
                    self._last_episode_init.encounter_setup
                ),
            )
        return info

    def _episode_outcome_info(self) -> dict:
        assert self.combat is not None
        max_hp = max(self.combat.player.max_hp, 1)
        hp_remaining = self.combat.player.current_hp
        won = self.combat.player.is_alive and all(
            not enemy.is_alive for enemy in self.combat.enemies
        )
        return {
            "won": won,
            "hp_remaining": hp_remaining,
            "hp_ratio_end": hp_remaining / max_hp,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        reset_instance_counter()
        options = options or {}

        rng_seed = int(self.np_random.integers(0, INT_MAX_EXCLUSIVE))
        rng = Rng(rng_seed)

        if "episode_init" in options:
            self._apply_episode_init(options["episode_init"], rng_seed)
        else:
            stage = self._resolve_curriculum_stage()
            if stage is not None:
                from sts2_env.training.combat_curriculum import sample_episode_init

                episode_init = sample_episode_init(
                    self.np_random,
                    stage,
                    hard_start_fraction=self.hard_start_fraction,
                    character_ids=self._resolve_character_pool(),
                )
                self._apply_episode_init(episode_init, rng_seed)
            else:
                char_id = resolve_character_for_episode(
                    self.np_random,
                    self._resolve_character_pool(),
                )
                self._last_character_id = char_id
                self._last_episode_init = None
                char_cfg = get_character(char_id)

                deck = create_starting_deck(char_id)
                player_hp = (
                    self._fixed_player_hp
                    if self._fixed_player_hp is not None
                    else char_cfg.starting_hp
                )
                player_max_hp = (
                    self._fixed_player_max_hp
                    if self._fixed_player_max_hp is not None
                    else char_cfg.starting_hp
                )

                self.combat = CombatState(
                    player_hp=player_hp,
                    player_max_hp=player_max_hp,
                    deck=deck,
                    rng_seed=rng_seed,
                    character_id=char_id,
                    relics=[char_cfg.starting_relic],
                    ascension_level=self._ascension_level,
                )

                encounter_idx = int(self.np_random.integers(0, len(self.encounter_pool)))
                encounter_setup = self.encounter_pool[encounter_idx]
                encounter_setup(self.combat, rng)
                self.combat.start_combat()

        self._event_cursor = CombatEventCursor()
        self._combat_gross_hp_lost = 0

        obs = encode_observation(self.combat)
        return obs, self._build_episode_info()

    def step(self, action: int):
        assert self.combat is not None, "Must call reset() first"

        prev_hp = self.combat.player.current_hp
        prev_alive_count = len(self.combat.alive_enemies)
        if self.combat.pending_choice is not None:
            if action == ACTION_END_TURN:
                self.combat.resolve_pending_choice(None)
            else:
                self.combat.resolve_pending_choice(action - 1)
        else:
            if action == ACTION_END_TURN:
                self.combat.end_player_turn()
            elif is_potion_action(action):
                slot_idx, target_idx = action_to_potion_and_target(action)
                success = (
                    slot_idx is not None
                    and self.combat.use_potion(slot_idx, target_index=target_idx)
                )
                if not success:
                    logger.debug("Ignored invalid potion action %d", action)
            else:
                hand_idx, target_idx = action_to_card_and_target(action)
                success = hand_idx is not None and self.combat.play_card(hand_idx, target_idx)
                if not success:
                    logger.debug("Ignored invalid card action %d", action)

        self._combat_gross_hp_lost += max(
            0, prev_hp - self.combat.player.current_hp,
        )
        obs = encode_observation(self.combat)
        reward, self._event_cursor = compute_reward(
            self.combat,
            prev_hp,
            reward_shaping=self.reward_shaping,
            reward_config=self._reward_config,
            event_cursor=self._event_cursor,
            prev_alive_count=prev_alive_count,
            combat_gross_hp_lost=self._combat_gross_hp_lost,
        )
        terminated = self.combat.is_over
        truncated = self.combat.turn_count > self.max_turns
        info = self._build_episode_info()
        if terminated or truncated:
            info.update(self._episode_outcome_info())

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return the current action mask (for sb3-contrib MaskablePPO)."""
        if self.combat is None:
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
            mask[0] = 1
            return mask
        return get_action_mask(self.combat)

    def render(self):
        if self.render_mode == "ansi" and self.combat is not None:
            return str(self.combat)
        return None

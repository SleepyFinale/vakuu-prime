"""SB3 callbacks for combat curriculum gate evaluation and auto-promotion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from sts2_env.training.combat_curriculum import CombatCurriculumStage, stage_at_index
from sts2_env.training.curriculum_env import write_curriculum_state


class CombatCurriculumEvalCallback(BaseCallback):
    """Evaluate promotion gates and optionally advance curriculum stages."""

    def __init__(
        self,
        *,
        gate_env,
        stage_sequence: Sequence[CombatCurriculumStage],
        initial_stage_index: int,
        output_dir: str | Path,
        eval_freq: int,
        n_eval_episodes: int,
        auto_promote: bool = False,
        character_ids: tuple[str, ...] | None = None,
        default_stage_budget: int = 500_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.gate_env = gate_env
        self.stage_sequence = list(stage_sequence)
        self.stage_index = initial_stage_index
        self.output_dir = Path(output_dir)
        self.eval_freq = max(eval_freq, 1)
        self.n_eval_episodes = n_eval_episodes
        self.auto_promote = auto_promote
        self.character_ids = character_ids
        self.default_stage_budget = default_stage_budget
        self._consecutive_passes = 0
        self._stage_enter_timestep = 0
        self._stage_step_deltas: list[int] = []

    def _on_training_start(self) -> None:
        self._stage_enter_timestep = self.num_timesteps

    def _current_stage(self) -> CombatCurriculumStage:
        return stage_at_index(self.stage_index, self.stage_sequence)

    def _run_gate_eval(self, model) -> tuple[float, float]:
        wins = 0
        hp_ratios: list[float] = []
        for episode in range(self.n_eval_episodes):
            obs, _info = self.gate_env.reset(seed=episode + 50_000)
            done = False
            last_info: dict[str, Any] = {}
            while not done:
                masks = self.gate_env.action_masks()
                action, _ = model.predict(
                    obs, action_masks=masks, deterministic=True
                )
                obs, _reward, terminated, truncated, last_info = self.gate_env.step(
                    int(action)
                )
                done = terminated or truncated
            if last_info.get("won"):
                wins += 1
            if "hp_ratio_end" in last_info:
                hp_ratios.append(float(last_info["hp_ratio_end"]))
        win_rate = wins / max(self.n_eval_episodes, 1)
        avg_hp = float(np.mean(hp_ratios)) if hp_ratios else 0.0
        return win_rate, avg_hp

    def _rebuild_gate_env(self) -> None:
        stage = self._current_stage()
        self.gate_env = build_gate_eval_env(
            stage,
            character_ids=self.character_ids,
        )

    def _stall_limit(self, gate) -> int:
        budget = gate.step_budget or self.default_stage_budget
        if self._stage_step_deltas:
            median_budget = int(np.median(self._stage_step_deltas))
        else:
            median_budget = budget
        return int(gate.force_promote_multiplier * median_budget)

    def _promote_stage(self, *, force: bool = False) -> bool:
        if self.stage_index + 1 >= len(self.stage_sequence):
            return False

        leaving_stage = self._current_stage()
        steps_on_stage = self.num_timesteps - self._stage_enter_timestep
        stall_limit = (
            self._stall_limit(leaving_stage.gate)
            if force and leaving_stage.gate is not None
            else 0
        )
        if steps_on_stage > 0:
            self._stage_step_deltas.append(steps_on_stage)

        self.stage_index += 1
        stage = self._current_stage()
        write_curriculum_state(
            self.output_dir,
            stage_index=self.stage_index,
            stage_name=stage.name,
        )
        checkpoint = self.output_dir / f"curriculum_stage_{self.stage_index}.zip"
        self.model.save(str(checkpoint))

        if force:
            self.logger.record("curriculum/forced_promotion", 1.0)
            self.logger.record("curriculum/steps_on_stage", float(steps_on_stage))
            self.logger.record("curriculum/stall_limit", float(stall_limit))
            if self.verbose:
                print(
                    f"\nCurriculum force-promoted to stage {self.stage_index}: "
                    f"{stage.name} after {steps_on_stage} steps "
                    f"(stall limit {stall_limit}; saved {checkpoint})"
                )
        elif self.verbose:
            print(
                f"\nCurriculum promoted to stage {self.stage_index}: "
                f"{stage.name} (saved {checkpoint})"
            )

        self._consecutive_passes = 0
        self._stage_enter_timestep = self.num_timesteps
        self._rebuild_gate_env()
        return True

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        stage = self._current_stage()
        win_rate, avg_hp = self._run_gate_eval(self.model)
        self.logger.record("curriculum/stage_index", self.stage_index)
        self.logger.record("curriculum/gate_win_rate", win_rate)
        self.logger.record("curriculum/gate_hp_ratio", avg_hp)

        gate = stage.gate
        if gate is None:
            return True

        self.logger.record("curriculum/gate_min_win_rate", gate.min_win_rate)
        self.logger.record("curriculum/gate_min_hp_ratio", gate.min_avg_hp_ratio)

        passed = (
            win_rate >= gate.min_win_rate
            and avg_hp >= gate.min_avg_hp_ratio
        )
        if passed:
            self._consecutive_passes += 1
        else:
            self._consecutive_passes = 0

        if (
            self.auto_promote
            and passed
            and self._consecutive_passes >= gate.consecutive_passes
        ):
            self._promote_stage()
            return True

        if self.auto_promote and not passed:
            steps_on_stage = self.num_timesteps - self._stage_enter_timestep
            stall_limit = self._stall_limit(gate)
            self.logger.record("curriculum/steps_on_stage", float(steps_on_stage))
            self.logger.record("curriculum/stall_limit", float(stall_limit))
            if steps_on_stage >= stall_limit:
                self._promote_stage(force=True)

        return True


def build_gate_eval_env(
    stage: CombatCurriculumStage,
    *,
    character_ids: tuple[str, ...] | None = None,
):
    """Build a sparse-reward env that samples only gate encounters."""
    from sb3_contrib.common.wrappers import ActionMasker
    from sts2_env.gym_env.combat_env import STS2CombatEnv
    from sts2_env.training.combat_curriculum import CombatCurriculumStage as Stage

    chars = character_ids or stage.character_ids
    gate_stage = Stage(
        name=f"{stage.name}_gate",
        encounter_pool=stage.gate_encounters,
        character_ids=chars,
        deck_templates=("starter",),
        hard_start_fraction=0.0,
        gate=None,
    )

    def mask_fn(env):
        return env.action_masks()

    env = STS2CombatEnv(
        character_ids=chars,
        curriculum_stage=gate_stage,
        reward_shaping=False,
    )
    return ActionMasker(env, mask_fn)

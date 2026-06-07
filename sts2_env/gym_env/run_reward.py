"""Run-level reward shaping for full-run RL training.

Dense progress bonuses (floor, combat clear, kills) accumulate over long
episodes.  Defaults are tuned so a maximal ~50-floor run stays under ~0.9
total progress shaping — keeping terminal +1 / -1 dominant and ensuring
long deaths cannot outscore short wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sts2_env.gym_env.reward_shaping import (
    CombatMicroRewardConfig,
    HpShapingConfig,
    compute_hp_loss_penalty,
)
from sts2_env.run.run_manager import RunManager

REWARD_WIN = 1.0
REWARD_DEATH = -1.0


def _default_run_micro_config() -> CombatMicroRewardConfig:
    return CombatMicroRewardConfig(kill_scale=0.003)


@dataclass(frozen=True)
class RunRewardConfig:
    """Tunable shaping scales (cumulative progress kept below terminal +/-1)."""

    floor_bonus: float = 0.01
    combat_clear_bonus: float = 0.005
    flawless_combat_bonus: float = 0.003
    win_hp_bonus_scale: float = 0.15
    hp: HpShapingConfig = field(default_factory=HpShapingConfig)
    micro: CombatMicroRewardConfig = field(default_factory=_default_run_micro_config)

    # Backward-compatible aliases for linear HP fields (deprecated).
    @property
    def hp_loss_penalty_scale(self) -> float:
        return self.hp.penalty_scale

    @property
    def max_hp_loss_penalty(self) -> float:
        return self.hp.max_penalty


@dataclass
class RunRewardSnapshot:
    """Run metrics captured before/after a step for shaping."""

    total_floor: int
    hp_ratio: float
    max_hp: int
    phase: str
    combat_active: bool
    last_combat_won: bool | None = None


def _hp_ratio(player) -> float:
    return player.current_hp / max(player.max_hp, 1)


def snapshot_from_manager(mgr: RunManager) -> RunRewardSnapshot:
    """Build a snapshot from the current RunManager state."""
    rs = mgr.run_state
    combat = mgr.get_combat_state()
    combat_active = (
        mgr.phase == RunManager.PHASE_COMBAT
        and combat is not None
        and not combat.is_over
    )
    last_combat_won: bool | None = None
    if combat is not None and combat.is_over:
        last_combat_won = combat.player_won
    return RunRewardSnapshot(
        total_floor=rs.total_floor,
        hp_ratio=_hp_ratio(rs.player),
        max_hp=rs.player.max_hp,
        phase=mgr.phase,
        combat_active=combat_active,
        last_combat_won=last_combat_won,
    )


def compute_run_shaping(
    prev: RunRewardSnapshot,
    curr: RunRewardSnapshot,
    config: RunRewardConfig,
    *,
    combat_gross_hp_lost: int | None = None,
) -> float:
    """Compute dense shaping reward between two snapshots."""
    reward = 0.0

    floor_delta = curr.total_floor - prev.total_floor
    if floor_delta > 0:
        reward += config.floor_bonus * floor_delta

    left_combat = (
        prev.phase == RunManager.PHASE_COMBAT
        and curr.phase != RunManager.PHASE_COMBAT
    )
    if left_combat and curr.last_combat_won is True:
        reward += config.combat_clear_bonus
        if combat_gross_hp_lost is not None and combat_gross_hp_lost <= 0:
            reward += config.flawless_combat_bonus
        hp_lost = max(0, int(round((prev.hp_ratio - curr.hp_ratio) * prev.max_hp)))
        if hp_lost > 0:
            reward -= compute_hp_loss_penalty(
                hp_lost,
                prev.max_hp,
                prev.hp_ratio,
                config.hp,
            )

    return reward


def compute_run_terminal_reward(
    *,
    player_won: bool,
    hp_ratio: float,
    config: RunRewardConfig,
    shaping_enabled: bool,
) -> float:
    """Terminal run reward (+1 / -1, plus optional HP efficiency bonus on win)."""
    if not player_won:
        return REWARD_DEATH
    if not shaping_enabled:
        return REWARD_WIN
    clamped_hp = max(0.0, min(1.0, hp_ratio))
    return REWARD_WIN + config.win_hp_bonus_scale * clamped_hp


@dataclass(frozen=True)
class NavigatorRewardConfig(RunRewardConfig):
    """Run shaping plus combat-critic draft signals for the Navigator agent."""

    draft_value_scale: float = 0.1
    deck_value_scale: float = 0.0


def compute_draft_value_shaping(
    draft_delta: float,
    config: NavigatorRewardConfig,
) -> float:
    """Shaping from combat critic delta-V on a card pick."""
    return config.draft_value_scale * draft_delta


def compute_navigator_shaping(
    prev: RunRewardSnapshot,
    curr: RunRewardSnapshot,
    config: NavigatorRewardConfig,
    *,
    draft_delta: float = 0.0,
    combat_gross_hp_lost: int | None = None,
) -> float:
    """Macro run shaping plus optional combat-value draft shaping."""
    reward = compute_run_shaping(
        prev, curr, config, combat_gross_hp_lost=combat_gross_hp_lost,
    )
    if draft_delta != 0.0:
        reward += compute_draft_value_shaping(draft_delta, config)
    return reward

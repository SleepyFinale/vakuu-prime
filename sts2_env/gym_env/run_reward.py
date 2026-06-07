"""Run-level reward shaping for full-run RL training."""

from __future__ import annotations

from dataclasses import dataclass, field

from sts2_env.gym_env.reward_shaping import (
    CombatMicroRewardConfig,
    HpShapingConfig,
    compute_hp_loss_penalty,
)
from sts2_env.run.run_manager import RunManager


@dataclass(frozen=True)
class RunRewardConfig:
    """Tunable shaping scales (kept well below terminal +/-1)."""

    floor_bonus: float = 0.05
    combat_clear_bonus: float = 0.1
    hp: HpShapingConfig = field(default_factory=HpShapingConfig)
    micro: CombatMicroRewardConfig = field(default_factory=CombatMicroRewardConfig)

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
        hp_lost = max(0, int(round((prev.hp_ratio - curr.hp_ratio) * prev.max_hp)))
        if hp_lost > 0:
            reward -= compute_hp_loss_penalty(
                hp_lost,
                prev.max_hp,
                prev.hp_ratio,
                config.hp,
            )

    return reward

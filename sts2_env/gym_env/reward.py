"""Reward calculation."""

from __future__ import annotations

from sts2_env.core.combat import CombatState
from sts2_env.gym_env.reward_shaping import (
    CombatEventCursor,
    CombatRewardConfig,
    compute_combat_hp_step_penalty,
    compute_combat_step_shaping,
)


def compute_reward(
    combat: CombatState,
    prev_hp: int,
    *,
    reward_shaping: bool = False,
    reward_config: CombatRewardConfig | None = None,
    event_cursor: CombatEventCursor | None = None,
) -> tuple[float, CombatEventCursor | None]:
    """Compute step reward.

    Sparse reward: +1 for win, -1 for loss, 0 otherwise.
    Optional shaping: non-linear HP penalty and combat micro-rewards.
    """
    if combat.is_over:
        return (1.0 if combat.player_won else -1.0), event_cursor

    reward = 0.0
    new_cursor = event_cursor

    if reward_shaping:
        config = reward_config or CombatRewardConfig()
        cursor = event_cursor or CombatEventCursor()
        micro_reward, new_cursor = compute_combat_step_shaping(
            combat, cursor, config.micro,
        )
        reward += micro_reward
        reward -= compute_combat_hp_step_penalty(prev_hp, combat, config.hp)

    return reward, new_cursor

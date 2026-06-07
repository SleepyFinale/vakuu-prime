"""Reward calculation."""

from __future__ import annotations

from sts2_env.core.combat import CombatState
from sts2_env.gym_env.reward_shaping import (
    CombatEventCursor,
    CombatRewardConfig,
    compute_combat_flawless_bonus,
    compute_combat_hp_step_penalty,
    compute_combat_kill_reward,
    compute_combat_step_shaping,
)


def compute_reward(
    combat: CombatState,
    prev_hp: int,
    *,
    reward_shaping: bool = False,
    reward_config: CombatRewardConfig | None = None,
    event_cursor: CombatEventCursor | None = None,
    prev_alive_count: int | None = None,
    combat_gross_hp_lost: int = 0,
) -> tuple[float, CombatEventCursor | None]:
    """Compute step reward.

    Sparse reward: +1 for win, -1 for loss, 0 otherwise.
    Optional shaping: kill credit, non-linear HP penalty, and combat micro-rewards.
    """
    config = reward_config or CombatRewardConfig()
    new_cursor = event_cursor

    if combat.is_over:
        reward = 1.0 if combat.player_won else -1.0
        if reward_shaping and combat.player_won:
            if prev_alive_count is not None:
                reward += compute_combat_kill_reward(
                    prev_alive_count, combat, config.micro,
                )
            reward += compute_combat_flawless_bonus(
                player_won=True,
                gross_hp_lost=combat_gross_hp_lost,
                bonus=config.flawless_bonus,
            )
        return reward, event_cursor

    reward = 0.0

    if reward_shaping:
        cursor = event_cursor or CombatEventCursor()
        if prev_alive_count is not None:
            reward += compute_combat_kill_reward(
                prev_alive_count, combat, config.micro,
            )
        micro_reward, new_cursor = compute_combat_step_shaping(
            combat, cursor, config.micro,
        )
        reward += micro_reward
        reward -= compute_combat_hp_step_penalty(prev_hp, combat, config.hp)

    return reward, new_cursor

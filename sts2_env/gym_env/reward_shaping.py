"""Shared reward shaping for combat and full-run training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from sts2_env.core.combat import CombatState
from sts2_env.core.enums import CombatSide, PowerId, ValueProp


@dataclass(frozen=True)
class HpShapingConfig:
    """Non-linear HP loss penalty scales (kept well below terminal +/-1)."""

    penalty_scale: float = 0.2
    max_penalty: float = 0.2
    steepness: float = 3.0


@dataclass(frozen=True)
class CombatMicroRewardConfig:
    """Per-step micro-rewards for key combat mechanics."""

    vulnerable_scale: float = 0.02
    weak_scale: float = 0.02
    block_scale: float = 0.001
    kill_scale: float = 0.05
    max_step_bonus: float = 0.05


@dataclass(frozen=True)
class CombatRewardConfig:
    """Combined combat shaping configuration."""

    hp: HpShapingConfig = field(default_factory=HpShapingConfig)
    micro: CombatMicroRewardConfig = field(default_factory=CombatMicroRewardConfig)
    flawless_bonus: float = 0.1


@dataclass
class CombatEventCursor:
    """Index into combat-level event logs for step-delta shaping."""

    power_events: int = 0
    damage_events: int = 0


def hp_marginal_weight(hp_ratio: float, steepness: float) -> float:
    """Higher weight when HP is low — same damage costs more near death."""
    ratio = max(0.0, min(1.0, hp_ratio))
    return math.exp(steepness * (1.0 - ratio))


def compute_hp_loss_penalty(
    hp_lost: int,
    max_hp: int,
    hp_ratio_before: float,
    config: HpShapingConfig,
) -> float:
    """Penalize HP loss with exponential marginal weighting."""
    if hp_lost <= 0:
        return 0.0
    relative = hp_lost / max(max_hp, 1)
    weight = hp_marginal_weight(hp_ratio_before, config.steepness)
    return min(relative * config.penalty_scale * weight, config.max_penalty)


def _is_player_side(creature) -> bool:
    return creature.side == CombatSide.PLAYER or getattr(creature, "is_player", False)


def _is_enemy_side(creature) -> bool:
    return creature.side == CombatSide.ENEMY and not getattr(creature, "is_player", False)


def _is_enemy_attack(props: ValueProp) -> bool:
    return props.is_powered_attack() or props.is_card_or_monster_move()


def debuff_stack_marginal(existing: int, amount: int, *, exponent: float = 0.6) -> float:
    """Sublinear marginal value for stacking debuffs on a target."""
    if amount <= 0:
        return 0.0
    return max(0.0, (existing + amount) ** exponent - existing ** exponent)


def compute_combat_step_shaping(
    combat: CombatState,
    cursor: CombatEventCursor,
    config: CombatMicroRewardConfig,
) -> tuple[float, CombatEventCursor]:
    """Compute micro-rewards from new combat events since the last cursor."""
    reward = 0.0

    power_events = combat._power_events_combat
    event_slice = power_events[cursor.power_events:]

    slice_delta: dict[tuple[object, PowerId], int] = {}
    for target, power_id, amount, applier in event_slice:
        if amount <= 0:
            continue
        if applier is None or not _is_player_side(applier):
            continue
        if not _is_enemy_side(target):
            continue
        if power_id not in (PowerId.VULNERABLE, PowerId.WEAK):
            continue
        key = (target, power_id)
        slice_delta[key] = slice_delta.get(key, 0) + amount

    stack_before: dict[tuple[object, PowerId], int] = {}
    for target, power_id, amount, applier in event_slice:
        if amount <= 0:
            continue
        if applier is None or not _is_player_side(applier):
            continue
        if not _is_enemy_side(target):
            continue
        if power_id == PowerId.VULNERABLE:
            scale = config.vulnerable_scale
        elif power_id == PowerId.WEAK:
            scale = config.weak_scale
        else:
            continue

        key = (target, power_id)
        if key not in stack_before:
            stack_before[key] = target.get_power_amount(power_id) - slice_delta[key]
        existing = stack_before[key]
        reward += debuff_stack_marginal(existing, amount) * scale
        stack_before[key] = existing + amount

    damage_events = combat._damage_events_combat
    player = combat.primary_player
    for dealer, target, props, _, blocked in damage_events[cursor.damage_events:]:
        if target is not player or blocked <= 0:
            continue
        if dealer is None or not _is_enemy_side(dealer):
            continue
        if not _is_enemy_attack(props):
            continue
        reward += blocked * config.block_scale

    reward = min(reward, config.max_step_bonus)

    new_cursor = CombatEventCursor(
        power_events=len(power_events),
        damage_events=len(damage_events),
    )
    return reward, new_cursor


def compute_combat_kill_reward(
    prev_alive_count: int,
    combat: CombatState,
    config: CombatMicroRewardConfig,
) -> float:
    """Reward for enemies killed during a single env step."""
    kills = max(0, prev_alive_count - len(combat.alive_enemies))
    return kills * config.kill_scale


def compute_combat_hp_step_penalty(
    prev_hp: int,
    combat: CombatState,
    config: HpShapingConfig,
) -> float:
    """Non-linear penalty for HP lost during a single env step."""
    player = combat.primary_player
    hp_lost = max(0, prev_hp - player.current_hp)
    if hp_lost <= 0:
        return 0.0
    max_hp = max(player.max_hp, 1)
    hp_ratio_before = prev_hp / max_hp
    return compute_hp_loss_penalty(hp_lost, max_hp, hp_ratio_before, config)


def compute_combat_flawless_bonus(
    *,
    player_won: bool,
    gross_hp_lost: int,
    bonus: float,
) -> float:
    """Bonus for winning combat without losing any HP during the fight."""
    if player_won and gross_hp_lost <= 0 and bonus > 0:
        return bonus
    return 0.0

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
    max_step_bonus: float = 0.05


@dataclass(frozen=True)
class CombatRewardConfig:
    """Combined combat shaping configuration."""

    hp: HpShapingConfig = field(default_factory=HpShapingConfig)
    micro: CombatMicroRewardConfig = field(default_factory=CombatMicroRewardConfig)


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


def compute_combat_step_shaping(
    combat: CombatState,
    cursor: CombatEventCursor,
    config: CombatMicroRewardConfig,
) -> tuple[float, CombatEventCursor]:
    """Compute micro-rewards from new combat events since the last cursor."""
    reward = 0.0

    power_events = combat._power_events_combat
    for target, power_id, amount, applier in power_events[cursor.power_events:]:
        if amount <= 0:
            continue
        if applier is None or not _is_player_side(applier):
            continue
        if not _is_enemy_side(target):
            continue
        if power_id == PowerId.VULNERABLE:
            reward += amount * config.vulnerable_scale
        elif power_id == PowerId.WEAK:
            reward += amount * config.weak_scale

    damage_events = combat._damage_events_combat
    player = combat.primary_player
    for dealer, target, props, unblocked, blocked in damage_events[cursor.damage_events:]:
        del unblocked
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

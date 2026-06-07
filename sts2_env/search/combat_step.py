"""Apply one combat action without Gym / reward overhead."""

from __future__ import annotations

from sts2_env.core.combat import CombatState
from sts2_env.core.constants import ACTION_END_TURN
from sts2_env.gym_env.action_space import (
    action_to_card_and_target,
    action_to_potion_and_target,
    is_potion_action,
)


def apply_combat_action(combat: CombatState, action: int) -> bool:
    """Apply *action* to *combat* in place.  Returns True if the action had effect."""
    if combat.pending_choice is not None:
        if action == ACTION_END_TURN:
            combat.resolve_pending_choice(None)
            return True
        combat.resolve_pending_choice(action - 1)
        return True

    if action == ACTION_END_TURN:
        combat.end_player_turn()
        return True

    if is_potion_action(action):
        slot_idx, target_idx = action_to_potion_and_target(action)
        if slot_idx is None:
            return False
        return combat.use_potion(slot_idx, target_index=target_idx)

    hand_idx, target_idx = action_to_card_and_target(action)
    if hand_idx is None:
        return False
    return combat.play_card(hand_idx, target_idx)

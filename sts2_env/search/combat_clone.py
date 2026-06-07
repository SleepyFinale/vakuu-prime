"""Deep-copy combat state for MCTS branching."""

from __future__ import annotations

import copy

from sts2_env.core.combat import CombatState


def clone_combat_state(combat: CombatState) -> CombatState:
    """Return an independent copy of *combat* suitable for simulated branching.

    Uses ``copy.deepcopy`` so card piles, creature powers, monster AI logs, and
    RNG counters stay aligned with the source snapshot.  Turn-bounded search only
    needs mid-combat snapshots; full-game hidden information is unchanged.
    """
    cloned = copy.deepcopy(combat)
    _rewire_combat_pointers(cloned)
    return cloned


def _rewire_combat_pointers(combat: CombatState) -> None:
    """Ensure every creature points at this combat after deepcopy."""
    combat._root_player.combat_state = combat
    for enemy in combat.enemies:
        enemy.combat_state = combat
    for ally in combat.allies:
        ally.combat_state = combat
    if combat.osty is not None:
        combat.osty.combat_state = combat
    for creature in combat._combat_player_state_by_creature:
        creature.combat_state = combat
    acting = combat._acting_player
    if acting is not None:
        acting.combat_state = combat

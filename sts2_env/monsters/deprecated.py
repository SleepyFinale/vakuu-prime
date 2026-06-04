"""Placeholder monsters kept for save compatibility."""

from __future__ import annotations

from sts2_env.core.creature import Creature
from sts2_env.core.rng import Rng
from sts2_env.monsters.intents import Intent, IntentType
from sts2_env.monsters.state_machine import MonsterAI, MoveState

DEPRECATED_MONSTER_ID = "DeprecatedMonster"
DEPRECATED_MONSTER_STUB_MOVE = "STUB"


def create_deprecated_monster(rng: Rng) -> tuple[Creature, MonsterAI]:
    """Matches DeprecatedMonster.cs: 0 HP stub with a hidden-style noop move."""
    creature = Creature(max_hp=0, current_hp=0, monster_id=DEPRECATED_MONSTER_ID)

    def stub_move(combat) -> None:
        return None

    states = {
        DEPRECATED_MONSTER_STUB_MOVE: MoveState(
            DEPRECATED_MONSTER_STUB_MOVE,
            stub_move,
            [Intent(IntentType.STUN)],
            follow_up_id=DEPRECATED_MONSTER_STUB_MOVE,
        ),
    }
    return creature, MonsterAI(states, DEPRECATED_MONSTER_STUB_MOVE)

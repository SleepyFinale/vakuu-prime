"""Encounter id -> setup function registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from sts2_env.core.combat import CombatState
    from sts2_env.core.rng import Rng

EncounterSetup = Callable[["CombatState", "Rng"], None]

from sts2_env.encounters.act1 import (
    setup_ceremonial_beast_boss,
    setup_the_kin_boss,
    setup_vantom_boss,
)
from sts2_env.encounters.act2 import (
    setup_kaiser_crab_boss,
    setup_knowledge_demon_boss,
    setup_the_insatiable_boss,
)
from sts2_env.encounters.act3 import (
    setup_doormaker_boss,
    setup_queen_boss,
    setup_test_subject_boss,
)
from sts2_env.encounters.act4 import (
    setup_lagavulin_matriarch_boss,
    setup_soul_fysh_boss,
    setup_waterfall_giant_boss,
)

BOSS_SETUP_BY_ID: dict[str, EncounterSetup] = {
    "VantomBoss": setup_vantom_boss,
    "CeremonialBeastBoss": setup_ceremonial_beast_boss,
    "TheKinBoss": setup_the_kin_boss,
    "TheInsatiableBoss": setup_the_insatiable_boss,
    "KnowledgeDemonBoss": setup_knowledge_demon_boss,
    "KaiserCrabBoss": setup_kaiser_crab_boss,
    "QueenBoss": setup_queen_boss,
    "TestSubjectBoss": setup_test_subject_boss,
    "DoormakerBoss": setup_doormaker_boss,
    "WaterfallGiantBoss": setup_waterfall_giant_boss,
    "SoulFyshBoss": setup_soul_fysh_boss,
    "LagavulinMatriarchBoss": setup_lagavulin_matriarch_boss,
}


def get_boss_setup(boss_id: str) -> EncounterSetup | None:
    return BOSS_SETUP_BY_ID.get(boss_id)

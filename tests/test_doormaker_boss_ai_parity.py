"""Doormaker boss phase AI parity with C# Doormaker.cs."""

import sts2_env.powers  # noqa: F401

from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import PowerId
from sts2_env.core.rng import Rng
from sts2_env.monsters.act3 import (
    DOORMAKER_DRAMATIC_OPEN_MOVE,
    DOORMAKER_GRASP_MOVE,
    DOORMAKER_HUNGER_MOVE,
    DOORMAKER_INFINITE_HP,
    DOORMAKER_SCRUTINY_MOVE,
    create_doormaker,
    create_door,
)
from sts2_env.monsters.intents import IntentType


def _make_combat(seed: int = 42) -> CombatState:
    return CombatState(
        player_hp=80,
        player_max_hp=80,
        deck=create_ironclad_starter_deck(),
        rng_seed=seed,
        character_id="Ironclad",
    )


class TestDoormakerBossAiParity:
    def test_after_added_to_combat_uses_infinite_hp_until_dramatic_open(self):
        combat = _make_combat()
        doormaker, ai = create_doormaker(Rng(1), ascension_level=9)
        combat.add_enemy(doormaker, ai)
        combat.start_combat()

        assert doormaker.max_hp == DOORMAKER_INFINITE_HP
        assert ai.doormaker_original_hp == 512
        assert ai.current_move.state_id == DOORMAKER_DRAMATIC_OPEN_MOVE

        ai.states[DOORMAKER_DRAMATIC_OPEN_MOVE].perform(combat)
        assert doormaker.max_hp == 512
        assert doormaker.get_power_amount(PowerId.HUNGER) == 1

    def test_phase_cycle_swaps_hunger_scrutiny_grasp(self):
        combat = _make_combat()
        doormaker, ai = create_doormaker(Rng(2))
        combat.add_enemy(doormaker, ai)
        combat.start_combat()
        ai.states[DOORMAKER_DRAMATIC_OPEN_MOVE].perform(combat)

        ai.states[DOORMAKER_HUNGER_MOVE].perform(combat)
        assert doormaker.get_power_amount(PowerId.SCRUTINY) == 1
        assert doormaker.get_power_amount(PowerId.HUNGER) == 0

        ai.states[DOORMAKER_SCRUTINY_MOVE].perform(combat)
        assert doormaker.get_power_amount(PowerId.GRASP) == 1

        ai.states[DOORMAKER_GRASP_MOVE].perform(combat)
        assert doormaker.get_power_amount(PowerId.STRENGTH) == 3
        assert doormaker.get_power_amount(PowerId.HUNGER) == 1
        assert doormaker.get_power_amount(PowerId.GRASP) == 0

    def test_dramatic_open_summon_intent_and_clears_existing_powers(self):
        combat = _make_combat()
        doormaker, ai = create_doormaker(Rng(3))
        combat.add_enemy(doormaker, ai)
        combat.start_combat()
        combat.apply_power_to(doormaker, PowerId.STRENGTH, 2, applier=doormaker)

        intent = ai.states[DOORMAKER_DRAMATIC_OPEN_MOVE].intents[0]
        assert intent.intent_type == IntentType.SUMMON

        ai.states[DOORMAKER_DRAMATIC_OPEN_MOVE].perform(combat)
        assert doormaker.get_power_amount(PowerId.STRENGTH) == 0
        assert doormaker.get_power_amount(PowerId.HUNGER) == 1

    def test_spawned_doormaker_gets_infinite_hp_via_add_enemy(self):
        combat = _make_combat(44)
        door, door_ai = create_door(Rng(44))
        combat.add_enemy(door, door_ai)
        combat.start_combat()
        assert not combat.is_over
        assert combat.kill_creature(door)

        doormaker = next(e for e in combat.enemies if e.monster_id == "DOORMAKER")
        assert doormaker.max_hp == DOORMAKER_INFINITE_HP

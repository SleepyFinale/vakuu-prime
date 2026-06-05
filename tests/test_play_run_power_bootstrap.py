"""Power registration via combat import (play_run / RunManager path).

Does not import sts2_env.powers at module level; registration must come from
sts2_env.core.combat importing the powers package.
"""

from sts2_env.cards.ironclad_basic import create_ironclad_starter_deck
from sts2_env.core.combat import CombatState
from sts2_env.core.creature import get_power_class
from sts2_env.core.enums import PowerId
from sts2_env.core.rng import Rng
from sts2_env.encounters.act4 import setup_cultists_normal
from sts2_env.powers.turn_effects import RitualPower


def test_lazy_power_registry_registers_ritual_on_first_lookup():
    assert get_power_class(PowerId.RITUAL) is RitualPower


def test_cultists_gain_ritual_then_strength_over_full_turns():
    combat = CombatState(
        player_hp=80,
        player_max_hp=80,
        deck=create_ironclad_starter_deck(),
        rng_seed=61,
    )
    setup_cultists_normal(combat, Rng(61))
    combat.start_combat()

    calcified = combat.enemies[0]
    damp = combat.enemies[1]

    combat.end_player_turn()
    assert calcified.get_power_amount(PowerId.RITUAL) == 2
    assert damp.get_power_amount(PowerId.RITUAL) == 5
    assert calcified.get_power_amount(PowerId.STRENGTH) == 0
    assert damp.get_power_amount(PowerId.STRENGTH) == 0

    combat.end_player_turn()
    assert calcified.get_power_amount(PowerId.STRENGTH) == 2
    assert damp.get_power_amount(PowerId.STRENGTH) == 5

    combat.end_player_turn()
    assert calcified.get_power_amount(PowerId.STRENGTH) == 4
    assert damp.get_power_amount(PowerId.STRENGTH) == 10

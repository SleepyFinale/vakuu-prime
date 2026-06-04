"""Parity tests for Doormaker phase powers and related debuffs."""

import sts2_env.powers  # noqa: F401

import pytest

from sts2_env.cards.ironclad_basic import make_defend_ironclad, make_strike_ironclad
from sts2_env.cards.factory import create_card
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import CardId, CardType, CombatSide, PowerId
from sts2_env.core.hooks import fire_after_turn_end, should_draw
from sts2_env.core.rng import Rng
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.monsters.act1_weak import create_shrinker_beetle


def _make_combat(seed: int = 42) -> CombatState:
    combat = CombatState(
        player_hp=80,
        player_max_hp=80,
        deck=create_ironclad_starter_deck(),
        rng_seed=seed,
        character_id="Ironclad",
    )
    creature, ai = create_shrinker_beetle(Rng(seed))
    combat.add_enemy(creature, ai)
    combat.start_combat()
    return combat


class TestDoormakerPhasePowersParity:
    def test_borrowed_time_power_increases_owner_card_cost(self):
        """BorrowedTimePower adds Amount to the owner's card costs until turn end."""
        combat = _make_combat()
        card = make_strike_ironclad()
        card.owner = combat.player
        combat.apply_power_to(combat.player, PowerId.BORROWED_TIME, 2, applier=combat.player)

        assert combat.modified_card_cost(combat.player, card) == card.cost + 2

        fire_after_turn_end(CombatSide.PLAYER, combat)
        assert combat.player.get_power_amount(PowerId.BORROWED_TIME) == 0

    def test_no_energy_gain_power_blocks_gain_energy(self):
        """NoEnergyGainPower zeroes energy gain for the owner."""
        combat = _make_combat()
        combat.apply_power_to(combat.player, PowerId.NO_ENERGY_GAIN, 1, applier=combat.player)
        combat.energy = 0

        combat.gain_energy(combat.player, 3)
        assert combat.energy == 0

        fire_after_turn_end(CombatSide.PLAYER, combat)
        assert combat.player.get_power_amount(PowerId.NO_ENERGY_GAIN) == 0
        combat.gain_energy(combat.player, 2)
        assert combat.energy == 2

    def test_scrutiny_power_blocks_non_hand_draw(self):
        """ScrutinyPower blocks mid-turn draws but allows opening hand draw."""
        combat = _make_combat()
        combat.apply_power_to(combat.enemies[0], PowerId.SCRUTINY, 1, applier=combat.enemies[0])

        assert should_draw(combat, combat.player, from_hand_draw=True) is True
        assert should_draw(combat, combat.player, from_hand_draw=False) is False

    def test_grasp_power_afflicts_weighted_on_apply_and_new_cards(self):
        """GraspPower afflicts existing and newly entered player cards with weighted."""
        combat = _make_combat()
        strike = make_strike_ironclad()
        strike.owner = combat.player
        combat.hand = [strike]
        combat.apply_power_to(combat.enemies[0], PowerId.GRASP, 1, applier=combat.enemies[0])

        assert strike.has_affliction("weighted")

        new_card = make_defend_ironclad()
        new_card.owner = combat.player
        combat.discard_pile.append(new_card)
        combat._apply_card_after_card_entered_combat(new_card, combat.player)
        assert new_card.has_affliction("weighted")

    def test_grasp_power_clears_weighted_on_remove(self):
        """GraspPower clears weighted afflictions when removed."""
        combat = _make_combat()
        strike = make_strike_ironclad()
        strike.owner = combat.player
        combat.hand = [strike]
        combat.apply_power_to(combat.enemies[0], PowerId.GRASP, 1, applier=combat.enemies[0])
        combat._remove_power(combat.enemies[0], PowerId.GRASP)

        assert not strike.has_affliction("weighted")

    def test_hunger_power_afflicts_devoured_attacks_and_skills_only(self):
        """HungerPower afflicts Attack/Skill cards with devoured and adds exhaust."""
        combat = _make_combat()
        strike = make_strike_ironclad()
        strike.owner = combat.player
        power_card = create_card(CardId.INFLAME)
        power_card.owner = combat.player
        combat.hand = [strike, power_card]
        combat.apply_power_to(combat.enemies[0], PowerId.HUNGER, 1, applier=combat.enemies[0])

        assert strike.has_affliction("devoured")
        assert "exhaust" in strike.keywords
        assert not power_card.has_affliction("devoured")

    def test_hunger_power_clears_devoured_on_remove(self):
        """HungerPower removes devoured affliction and applied exhaust on cleanup."""
        combat = _make_combat()
        strike = make_strike_ironclad()
        strike.owner = combat.player
        combat.hand = [strike]
        combat.apply_power_to(combat.enemies[0], PowerId.HUNGER, 1, applier=combat.enemies[0])
        combat._remove_power(combat.enemies[0], PowerId.HUNGER)

        assert not strike.has_affliction("devoured")
        assert "exhaust" not in strike.keywords

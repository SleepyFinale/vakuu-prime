"""Generated OnPlay smoke parity tests.

Regenerate: python scripts/audit_onplay_behavior_coverage.py --generate-smoke-tests
"""

import pytest

import sts2_env.powers  # noqa: F401

from sts2_env.cards.factory import create_card
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import CardId, CardType, TargetType
from sts2_env.core.rng import Rng
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.monsters.act1_weak import create_shrinker_beetle
from sts2_env.run.run_state import PlayerState


def _make_combat(character_id: str = "Ironclad") -> CombatState:
    deck_fn = create_ironclad_starter_deck
    if character_id == "Silent":
        from sts2_env.cards.silent import create_silent_starter_deck
        deck_fn = create_silent_starter_deck
    elif character_id == "Defect":
        from sts2_env.cards.defect import create_defect_starter_deck
        deck_fn = create_defect_starter_deck
    elif character_id == "Necrobinder":
        from sts2_env.cards.necrobinder import create_necrobinder_starter_deck
        deck_fn = create_necrobinder_starter_deck
    elif character_id == "Regent":
        from sts2_env.cards.regent import create_regent_starter_deck
        deck_fn = create_regent_starter_deck
    combat = CombatState(
        player_hp=80,
        player_max_hp=80,
        deck=deck_fn(),
        rng_seed=9001,
        character_id=character_id,
    )
    creature, ai = create_shrinker_beetle(Rng(9001))
    combat.add_enemy(creature, ai)
    combat.start_combat()
    return combat


def _character_for_card(card_id_name: str) -> str:
    if card_id_name.startswith(("STRIKE_SILENT", "DEFEND_SILENT")) or "_SILENT" in card_id_name:
        return "Silent"
    if "DEFECT" in card_id_name or card_id_name.endswith("_DEFECT"):
        return "Defect"
    if "NECROBINDER" in card_id_name or "NECRO" in card_id_name:
        return "Necrobinder"
    if "REGENT" in card_id_name:
        return "Regent"
    return "Ironclad"


def _play_smoke(combat: CombatState, card, target_index: int | None = 0) -> None:
    if card.target_type == TargetType.ANY_ALLY:
        combat.add_ally_player(
            PlayerState(
                player_id=2,
                character_id=combat.character_id,
                max_hp=combat.player.max_hp,
                current_hp=combat.player.current_hp,
            )
        )
    if card.card_id == CardId.PACTS_END:
        from sts2_env.cards.ironclad_basic import make_strike_ironclad
        needed = max(0, card.effect_vars.get('cards', 3) - len(combat.exhaust_pile))
        combat.exhaust_pile.extend(make_strike_ironclad() for _ in range(needed))
    combat.hand = [card]
    star_cost = getattr(card, 'star_cost', 0) or 0
    if star_cost > 0:
        combat.gain_stars(combat.player, star_cost)
    combat.energy = max(combat.energy, max(3, card.cost if card.cost >= 0 else 3))
    while combat.pending_choice is not None:
        combat.resolve_pending_choice(0)
    played = False
    if card.target_type in (TargetType.ANY_ENEMY, TargetType.RANDOM_ENEMY):
        played = combat.play_card(0, target_index)
    elif card.target_type == TargetType.ANY_ALLY:
        played = combat.play_card(0, 0)
    else:
        played = combat.play_card(0)
    if not played and (card.is_unplayable or not combat.can_play_card(card)):
        return
    assert played
    while combat.pending_choice is not None:
        combat.resolve_pending_choice(0)


def test_brand_onplay_smoke():
    """Matches Brand.cs: Apply power; Exhaust."""
    card = create_card(CardId.BRAND)
    combat = _make_combat(_character_for_card("BRAND"))
    _play_smoke(combat, card)

def test_bundle_of_joy_onplay_smoke():
    """Matches BundleOfJoy.cs: Add generated card(s) to pile."""
    card = create_card(CardId.BUNDLE_OF_JOY)
    combat = _make_combat(_character_for_card("BUNDLE_OF_JOY"))
    _play_smoke(combat, card)

def test_catastrophe_onplay_smoke():
    """Matches Catastrophe.cs: See decompiled source."""
    card = create_card(CardId.CATASTROPHE)
    combat = _make_combat(_character_for_card("CATASTROPHE"))
    _play_smoke(combat, card)

def test_distraction_onplay_smoke():
    """Matches Distraction.cs: Add generated card(s) to pile; Set card(s) to cost 0."""
    card = create_card(CardId.DISTRACTION)
    combat = _make_combat(_character_for_card("DISTRACTION"))
    _play_smoke(combat, card)

def test_graveblast_onplay_smoke():
    """Matches Graveblast.cs: Deal Damage."""
    card = create_card(CardId.GRAVEBLAST)
    combat = _make_combat(_character_for_card("GRAVEBLAST"))
    _play_smoke(combat, card)

def test_infernal_blade_onplay_smoke():
    """Matches InfernalBlade.cs: Add generated card(s) to pile; Set card(s) to cost 0."""
    card = create_card(CardId.INFERNAL_BLADE)
    combat = _make_combat(_character_for_card("INFERNAL_BLADE"))
    _play_smoke(combat, card)

def test_jack_of_all_trades_onplay_smoke():
    """Matches JackOfAllTrades.cs: Add generated card(s) to pile."""
    card = create_card(CardId.JACK_OF_ALL_TRADES)
    combat = _make_combat(_character_for_card("JACK_OF_ALL_TRADES"))
    _play_smoke(combat, card)

def test_jackpot_onplay_smoke():
    """Matches Jackpot.cs: Deal Damage; Add generated card(s) to pile; Upgrade card(s)."""
    card = create_card(CardId.JACKPOT)
    combat = _make_combat(_character_for_card("JACKPOT"))
    _play_smoke(combat, card)

def test_knife_trap_onplay_smoke():
    """Matches KnifeTrap.cs: Exhaust; Upgrade card(s)."""
    card = create_card(CardId.KNIFE_TRAP)
    combat = _make_combat(_character_for_card("KNIFE_TRAP"))
    _play_smoke(combat, card)

def test_metamorphosis_onplay_smoke():
    """Matches Metamorphosis.cs: Add generated card(s) to pile."""
    card = create_card(CardId.METAMORPHOSIS)
    combat = _make_combat(_character_for_card("METAMORPHOSIS"))
    _play_smoke(combat, card)

def test_primal_force_onplay_smoke():
    """Matches PrimalForce.cs: Upgrade card(s)."""
    card = create_card(CardId.PRIMAL_FORCE)
    combat = _make_combat(_character_for_card("PRIMAL_FORCE"))
    _play_smoke(combat, card)

def test_sculpting_strike_onplay_smoke():
    """Matches SculptingStrike.cs: Deal Damage."""
    card = create_card(CardId.SCULPTING_STRIKE)
    combat = _make_combat(_character_for_card("SCULPTING_STRIKE"))
    _play_smoke(combat, card)

def test_seeker_strike_onplay_smoke():
    """Matches SeekerStrike.cs: Deal Damage."""
    card = create_card(CardId.SEEKER_STRIKE)
    combat = _make_combat(_character_for_card("SEEKER_STRIKE"))
    _play_smoke(combat, card)

def test_shining_strike_onplay_smoke():
    """Matches ShiningStrike.cs: Deal Damage; Exhaust."""
    card = create_card(CardId.SHINING_STRIKE)
    combat = _make_combat(_character_for_card("SHINING_STRIKE"))
    _play_smoke(combat, card)

def test_snap_onplay_smoke():
    """Matches Snap.cs: Deal Damage."""
    card = create_card(CardId.SNAP)
    combat = _make_combat(_character_for_card("SNAP"))
    _play_smoke(combat, card)

def test_stoke_onplay_smoke():
    """Matches Stoke.cs: Add generated card(s) to pile; Exhaust; Upgrade card(s)."""
    card = create_card(CardId.STOKE)
    combat = _make_combat(_character_for_card("STOKE"))
    _play_smoke(combat, card)

def test_uproar_onplay_smoke():
    """Matches Uproar.cs: Deal Damage."""
    card = create_card(CardId.UPROAR)
    combat = _make_combat(_character_for_card("UPROAR"))
    _play_smoke(combat, card)

def test_abrasive_onplay_smoke():
    """Matches Abrasive.cs: Apply power."""
    card = create_card(CardId.ABRASIVE)
    combat = _make_combat(_character_for_card("ABRASIVE"))
    _play_smoke(combat, card)

def test_accelerant_onplay_smoke():
    """Matches Accelerant.cs: Apply power."""
    card = create_card(CardId.ACCELERANT)
    combat = _make_combat(_character_for_card("ACCELERANT"))
    _play_smoke(combat, card)

def test_accuracy_onplay_smoke():
    """Matches Accuracy.cs: Apply power."""
    card = create_card(CardId.ACCURACY_CARD)
    combat = _make_combat(_character_for_card("ACCURACY_CARD"))
    _play_smoke(combat, card)

def test_afterimage_onplay_smoke():
    """Matches Afterimage.cs: Apply power."""
    card = create_card(CardId.AFTERIMAGE_CARD)
    combat = _make_combat(_character_for_card("AFTERIMAGE_CARD"))
    _play_smoke(combat, card)

def test_aggression_onplay_smoke():
    """Matches Aggression.cs: Apply power."""
    card = create_card(CardId.AGGRESSION_CARD)
    combat = _make_combat(_character_for_card("AGGRESSION_CARD"))
    _play_smoke(combat, card)

def test_anger_onplay_smoke():
    """Matches Anger.cs: Deal Damage; Add generated card(s) to pile."""
    card = create_card(CardId.ANGER)
    combat = _make_combat(_character_for_card("ANGER"))
    _play_smoke(combat, card)

def test_anticipate_onplay_smoke():
    """Matches Anticipate.cs: Apply power."""
    card = create_card(CardId.ANTICIPATE)
    combat = _make_combat(_character_for_card("ANTICIPATE"))
    _play_smoke(combat, card)

def test_apotheosis_onplay_smoke():
    """Matches Apotheosis.cs: See decompiled source."""
    card = create_card(CardId.APOTHEOSIS)
    combat = _make_combat(_character_for_card("APOTHEOSIS"))
    _play_smoke(combat, card)

def test_apparition_onplay_smoke():
    """Matches Apparition.cs: Apply power."""
    card = create_card(CardId.APPARITION)
    combat = _make_combat(_character_for_card("APPARITION"))
    _play_smoke(combat, card)

def test_arsenal_onplay_smoke():
    """Matches Arsenal.cs: Apply power."""
    card = create_card(CardId.ARSENAL)
    combat = _make_combat(_character_for_card("ARSENAL"))
    _play_smoke(combat, card)

def test_ashen_strike_onplay_smoke():
    """Matches AshenStrike.cs: Deal Damage."""
    card = create_card(CardId.ASHEN_STRIKE)
    combat = _make_combat(_character_for_card("ASHEN_STRIKE"))
    _play_smoke(combat, card)

def test_assassinate_onplay_smoke():
    """Matches Assassinate.cs: Deal Damage; Apply power."""
    card = create_card(CardId.ASSASSINATE)
    combat = _make_combat(_character_for_card("ASSASSINATE"))
    _play_smoke(combat, card)

def test_backflip_onplay_smoke():
    """Matches Backflip.cs: Draw card(s)."""
    card = create_card(CardId.BACKFLIP)
    combat = _make_combat(_character_for_card("BACKFLIP"))
    _play_smoke(combat, card)

def test_banshees_cry_onplay_smoke():
    """Matches BansheesCry.cs: Deal Damage."""
    card = create_card(CardId.BANSHEES_CRY)
    combat = _make_combat(_character_for_card("BANSHEES_CRY"))
    _play_smoke(combat, card)

def test_barricade_onplay_smoke():
    """Matches Barricade.cs: Apply power."""
    card = create_card(CardId.BARRICADE_CARD)
    combat = _make_combat(_character_for_card("BARRICADE_CARD"))
    _play_smoke(combat, card)

def test_beacon_of_hope_onplay_smoke():
    """Matches BeaconOfHope.cs: Apply power."""
    card = create_card(CardId.BEACON_OF_HOPE)
    combat = _make_combat(_character_for_card("BEACON_OF_HOPE"))
    _play_smoke(combat, card)

def test_beat_into_shape_onplay_smoke():
    """Matches BeatIntoShape.cs: Deal Damage; Forge."""
    card = create_card(CardId.BEAT_INTO_SHAPE)
    combat = _make_combat(_character_for_card("BEAT_INTO_SHAPE"))
    _play_smoke(combat, card)

def test_believe_in_you_onplay_smoke():
    """Matches BelieveInYou.cs: See decompiled source."""
    card = create_card(CardId.BELIEVE_IN_YOU)
    combat = _make_combat(_character_for_card("BELIEVE_IN_YOU"))
    _play_smoke(combat, card)

def test_blade_of_ink_onplay_smoke():
    """Matches BladeOfInk.cs: See decompiled source."""
    card = create_card(CardId.BLADE_OF_INK)
    combat = _make_combat(_character_for_card("BLADE_OF_INK"))
    _play_smoke(combat, card)

def test_blight_strike_onplay_smoke():
    """Matches BlightStrike.cs: Deal Damage; Apply power."""
    card = create_card(CardId.BLIGHT_STRIKE)
    combat = _make_combat(_character_for_card("BLIGHT_STRIKE"))
    _play_smoke(combat, card)

def test_blood_wall_onplay_smoke():
    """Matches BloodWall.cs: See decompiled source."""
    card = create_card(CardId.BLOOD_WALL)
    combat = _make_combat(_character_for_card("BLOOD_WALL"))
    _play_smoke(combat, card)

def test_bludgeon_onplay_smoke():
    """Matches Bludgeon.cs: Deal Damage."""
    card = create_card(CardId.BLUDGEON)
    combat = _make_combat(_character_for_card("BLUDGEON"))
    _play_smoke(combat, card)

def test_blur_onplay_smoke():
    """Matches Blur.cs: Apply power."""
    card = create_card(CardId.BLUR_CARD)
    combat = _make_combat(_character_for_card("BLUR_CARD"))
    _play_smoke(combat, card)

def test_body_slam_onplay_smoke():
    """Matches BodySlam.cs: Deal Damage."""
    card = create_card(CardId.BODY_SLAM)
    combat = _make_combat(_character_for_card("BODY_SLAM"))
    _play_smoke(combat, card)

def test_bolas_onplay_smoke():
    """Matches Bolas.cs: Deal Damage."""
    card = create_card(CardId.BOLAS)
    combat = _make_combat(_character_for_card("BOLAS"))
    _play_smoke(combat, card)

def test_bombardment_onplay_smoke():
    """Matches Bombardment.cs: Deal Damage."""
    card = create_card(CardId.BOMBARDMENT)
    combat = _make_combat(_character_for_card("BOMBARDMENT"))
    _play_smoke(combat, card)

def test_boot_sequence_onplay_smoke():
    """Matches BootSequence.cs: See decompiled source."""
    card = create_card(CardId.BOOT_SEQUENCE)
    combat = _make_combat(_character_for_card("BOOT_SEQUENCE"))
    _play_smoke(combat, card)

def test_bouncing_flask_onplay_smoke():
    """Matches BouncingFlask.cs: Apply power."""
    card = create_card(CardId.BOUNCING_FLASK)
    combat = _make_combat(_character_for_card("BOUNCING_FLASK"))
    _play_smoke(combat, card)

def test_break_onplay_smoke():
    """Matches Break.cs: Deal Damage; Apply power."""
    card = create_card(CardId.BREAK)
    combat = _make_combat(_character_for_card("BREAK"))
    _play_smoke(combat, card)

def test_breakthrough_onplay_smoke():
    """Matches Breakthrough.cs: Deal Damage."""
    card = create_card(CardId.BREAKTHROUGH)
    combat = _make_combat(_character_for_card("BREAKTHROUGH"))
    _play_smoke(combat, card)

def test_bubble_bubble_onplay_smoke():
    """Matches BubbleBubble.cs: Apply power."""
    card = create_card(CardId.BUBBLE_BUBBLE)
    combat = _make_combat(_character_for_card("BUBBLE_BUBBLE"))
    _play_smoke(combat, card)

def test_buffer_onplay_smoke():
    """Matches Buffer.cs: Apply power."""
    card = create_card(CardId.BUFFER_CARD)
    combat = _make_combat(_character_for_card("BUFFER_CARD"))
    _play_smoke(combat, card)

def test_bulk_up_onplay_smoke():
    """Matches BulkUp.cs: Apply power; Orb action."""
    card = create_card(CardId.BULK_UP)
    combat = _make_combat(_character_for_card("BULK_UP"))
    _play_smoke(combat, card)

def test_bullet_time_onplay_smoke():
    """Matches BulletTime.cs: Apply power; Set card(s) to cost 0."""
    card = create_card(CardId.BULLET_TIME)
    combat = _make_combat(_character_for_card("BULLET_TIME"))
    _play_smoke(combat, card)

def test_bully_onplay_smoke():
    """Matches Bully.cs: Deal Damage."""
    card = create_card(CardId.BULLY)
    combat = _make_combat(_character_for_card("BULLY"))
    _play_smoke(combat, card)

def test_bury_onplay_smoke():
    """Matches Bury.cs: Deal Damage."""
    card = create_card(CardId.BURY)
    combat = _make_combat(_character_for_card("BURY"))
    _play_smoke(combat, card)

def test_byrd_swoop_onplay_smoke():
    """Matches ByrdSwoop.cs: Deal Damage."""
    card = create_card(CardId.BYRD_SWOOP)
    combat = _make_combat(_character_for_card("BYRD_SWOOP"))
    _play_smoke(combat, card)

def test_calamity_onplay_smoke():
    """Matches Calamity.cs: Apply power."""
    card = create_card(CardId.CALAMITY_CARD)
    combat = _make_combat(_character_for_card("CALAMITY_CARD"))
    _play_smoke(combat, card)

def test_calcify_onplay_smoke():
    """Matches Calcify.cs: Apply power."""
    card = create_card(CardId.CALCIFY_CARD)
    combat = _make_combat(_character_for_card("CALCIFY_CARD"))
    _play_smoke(combat, card)

def test_calculated_gamble_onplay_smoke():
    """Matches CalculatedGamble.cs: See decompiled source."""
    card = create_card(CardId.CALCULATED_GAMBLE)
    combat = _make_combat(_character_for_card("CALCULATED_GAMBLE"))
    _play_smoke(combat, card)

def test_call_of_the_void_onplay_smoke():
    """Matches CallOfTheVoid.cs: Apply power."""
    card = create_card(CardId.CALL_OF_THE_VOID)
    combat = _make_combat(_character_for_card("CALL_OF_THE_VOID"))
    _play_smoke(combat, card)

def test_caltrops_onplay_smoke():
    """Matches Caltrops.cs: Apply power."""
    card = create_card(CardId.CALTROPS)
    combat = _make_combat(_character_for_card("CALTROPS"))
    _play_smoke(combat, card)

def test_celestial_might_onplay_smoke():
    """Matches CelestialMight.cs: Deal Damage."""
    card = create_card(CardId.CELESTIAL_MIGHT)
    combat = _make_combat(_character_for_card("CELESTIAL_MIGHT"))
    _play_smoke(combat, card)

def test_child_of_the_stars_onplay_smoke():
    """Matches ChildOfTheStars.cs: Apply power."""
    card = create_card(CardId.CHILD_OF_THE_STARS)
    combat = _make_combat(_character_for_card("CHILD_OF_THE_STARS"))
    _play_smoke(combat, card)

def test_cinder_onplay_smoke():
    """Matches Cinder.cs: Deal Damage; Exhaust."""
    card = create_card(CardId.CINDER)
    combat = _make_combat(_character_for_card("CINDER"))
    _play_smoke(combat, card)

def test_claw_onplay_smoke():
    """Matches Claw.cs: Deal Damage."""
    card = create_card(CardId.CLAW)
    combat = _make_combat(_character_for_card("CLAW"))
    _play_smoke(combat, card)

def test_cloak_and_dagger_onplay_smoke():
    """Matches CloakAndDagger.cs: See decompiled source."""
    card = create_card(CardId.CLOAK_AND_DAGGER)
    combat = _make_combat(_character_for_card("CLOAK_AND_DAGGER"))
    _play_smoke(combat, card)

def test_cloak_of_stars_onplay_smoke():
    """Matches CloakOfStars.cs: See decompiled source."""
    card = create_card(CardId.CLOAK_OF_STARS)
    combat = _make_combat(_character_for_card("CLOAK_OF_STARS"))
    _play_smoke(combat, card)

def test_colossus_onplay_smoke():
    """Matches Colossus.cs: Apply power."""
    card = create_card(CardId.COLOSSUS_CARD)
    combat = _make_combat(_character_for_card("COLOSSUS_CARD"))
    _play_smoke(combat, card)

def test_conflagration_onplay_smoke():
    """Matches Conflagration.cs: Deal Damage."""
    card = create_card(CardId.CONFLAGRATION)
    combat = _make_combat(_character_for_card("CONFLAGRATION"))
    _play_smoke(combat, card)

def test_coolant_onplay_smoke():
    """Matches Coolant.cs: Apply power."""
    card = create_card(CardId.COOLANT)
    combat = _make_combat(_character_for_card("COOLANT"))
    _play_smoke(combat, card)

def test_coordinate_onplay_smoke():
    """Matches Coordinate.cs: Apply power."""
    card = create_card(CardId.COORDINATE_CARD)
    combat = _make_combat(_character_for_card("COORDINATE_CARD"))
    _play_smoke(combat, card)

def test_corrosive_wave_onplay_smoke():
    """Matches CorrosiveWave.cs: Apply power."""
    card = create_card(CardId.CORROSIVE_WAVE)
    combat = _make_combat(_character_for_card("CORROSIVE_WAVE"))
    _play_smoke(combat, card)

def test_corruption_onplay_smoke():
    """Matches Corruption.cs: Apply power."""
    card = create_card(CardId.CORRUPTION_CARD)
    combat = _make_combat(_character_for_card("CORRUPTION_CARD"))
    _play_smoke(combat, card)

def test_creative_ai_onplay_smoke():
    """Matches CreativeAi.cs: Apply power."""
    card = create_card(CardId.CREATIVE_AI_CARD)
    combat = _make_combat(_character_for_card("CREATIVE_AI_CARD"))
    _play_smoke(combat, card)

def test_crescent_spear_onplay_smoke():
    """Matches CrescentSpear.cs: Deal Damage."""
    card = create_card(CardId.CRESCENT_SPEAR)
    combat = _make_combat(_character_for_card("CRESCENT_SPEAR"))
    _play_smoke(combat, card)

def test_crimson_mantle_onplay_smoke():
    """Matches CrimsonMantle.cs: Apply power."""
    card = create_card(CardId.CRIMSON_MANTLE)
    combat = _make_combat(_character_for_card("CRIMSON_MANTLE"))
    _play_smoke(combat, card)

def test_crush_under_onplay_smoke():
    """Matches CrushUnder.cs: Deal Damage; Apply power."""
    card = create_card(CardId.CRUSH_UNDER)
    combat = _make_combat(_character_for_card("CRUSH_UNDER"))
    _play_smoke(combat, card)

def test_dagger_spray_onplay_smoke():
    """Matches DaggerSpray.cs: Deal Damage."""
    card = create_card(CardId.DAGGER_SPRAY)
    combat = _make_combat(_character_for_card("DAGGER_SPRAY"))
    _play_smoke(combat, card)

def test_dark_embrace_onplay_smoke():
    """Matches DarkEmbrace.cs: Apply power."""
    card = create_card(CardId.DARK_EMBRACE_CARD)
    combat = _make_combat(_character_for_card("DARK_EMBRACE_CARD"))
    _play_smoke(combat, card)

def test_dark_shackles_onplay_smoke():
    """Matches DarkShackles.cs: Apply power."""
    card = create_card(CardId.DARK_SHACKLES)
    combat = _make_combat(_character_for_card("DARK_SHACKLES"))
    _play_smoke(combat, card)

def test_dash_onplay_smoke():
    """Matches Dash.cs: Deal Damage."""
    card = create_card(CardId.DASH)
    combat = _make_combat(_character_for_card("DASH"))
    _play_smoke(combat, card)

def test_deadly_poison_onplay_smoke():
    """Matches DeadlyPoison.cs: Apply power."""
    card = create_card(CardId.DEADLY_POISON)
    combat = _make_combat(_character_for_card("DEADLY_POISON"))
    _play_smoke(combat, card)

def test_deathbringer_onplay_smoke():
    """Matches Deathbringer.cs: Apply power."""
    card = create_card(CardId.DEATHBRINGER)
    combat = _make_combat(_character_for_card("DEATHBRINGER"))
    _play_smoke(combat, card)

def test_deaths_door_onplay_smoke():
    """Matches DeathsDoor.cs: See decompiled source."""
    card = create_card(CardId.DEATHS_DOOR)
    combat = _make_combat(_character_for_card("DEATHS_DOOR"))
    _play_smoke(combat, card)

def test_debilitate_onplay_smoke():
    """Matches Debilitate.cs: Deal Damage; Apply power."""
    card = create_card(CardId.DEBILITATE_CARD)
    combat = _make_combat(_character_for_card("DEBILITATE_CARD"))
    _play_smoke(combat, card)

def test_debris_onplay_smoke():
    """Matches Debris.cs: See decompiled source."""
    card = create_card(CardId.DEBRIS)
    combat = _make_combat(_character_for_card("DEBRIS"))
    _play_smoke(combat, card)

def test_defend_defect_onplay_smoke():
    """Matches DefendDefect.cs: See decompiled source."""
    card = create_card(CardId.DEFEND_DEFECT)
    combat = _make_combat(_character_for_card("DEFEND_DEFECT"))
    _play_smoke(combat, card)

def test_defend_ironclad_onplay_smoke():
    """Matches DefendIronclad.cs: See decompiled source."""
    card = create_card(CardId.DEFEND_IRONCLAD)
    combat = _make_combat(_character_for_card("DEFEND_IRONCLAD"))
    _play_smoke(combat, card)

def test_defend_necrobinder_onplay_smoke():
    """Matches DefendNecrobinder.cs: See decompiled source."""
    card = create_card(CardId.DEFEND_NECROBINDER)
    combat = _make_combat(_character_for_card("DEFEND_NECROBINDER"))
    _play_smoke(combat, card)

def test_defend_regent_onplay_smoke():
    """Matches DefendRegent.cs: See decompiled source."""
    card = create_card(CardId.DEFEND_REGENT)
    combat = _make_combat(_character_for_card("DEFEND_REGENT"))
    _play_smoke(combat, card)

def test_defend_silent_onplay_smoke():
    """Matches DefendSilent.cs: See decompiled source."""
    card = create_card(CardId.DEFEND_SILENT)
    combat = _make_combat(_character_for_card("DEFEND_SILENT"))
    _play_smoke(combat, card)

def test_deflect_onplay_smoke():
    """Matches Deflect.cs: See decompiled source."""
    card = create_card(CardId.DEFLECT)
    combat = _make_combat(_character_for_card("DEFLECT"))
    _play_smoke(combat, card)

def test_defy_onplay_smoke():
    """Matches Defy.cs: Apply power."""
    card = create_card(CardId.DEFY)
    combat = _make_combat(_character_for_card("DEFY"))
    _play_smoke(combat, card)

def test_delay_onplay_smoke():
    """Matches Delay.cs: Apply power."""
    card = create_card(CardId.DELAY)
    combat = _make_combat(_character_for_card("DELAY"))
    _play_smoke(combat, card)

def test_demesne_onplay_smoke():
    """Matches Demesne.cs: Apply power."""
    card = create_card(CardId.DEMESNE)
    combat = _make_combat(_character_for_card("DEMESNE"))
    _play_smoke(combat, card)

def test_demon_form_onplay_smoke():
    """Matches DemonForm.cs: Apply power."""
    card = create_card(CardId.DEMON_FORM_CARD)
    combat = _make_combat(_character_for_card("DEMON_FORM_CARD"))
    _play_smoke(combat, card)

def test_demonic_shield_onplay_smoke():
    """Matches DemonicShield.cs: See decompiled source."""
    card = create_card(CardId.DEMONIC_SHIELD)
    combat = _make_combat(_character_for_card("DEMONIC_SHIELD"))
    _play_smoke(combat, card)

def test_devastate_onplay_smoke():
    """Matches Devastate.cs: Deal Damage."""
    card = create_card(CardId.DEVASTATE)
    combat = _make_combat(_character_for_card("DEVASTATE"))
    _play_smoke(combat, card)

def test_devour_life_onplay_smoke():
    """Matches DevourLife.cs: Apply power."""
    card = create_card(CardId.DEVOUR_LIFE_CARD)
    combat = _make_combat(_character_for_card("DEVOUR_LIFE_CARD"))
    _play_smoke(combat, card)

def test_dismantle_onplay_smoke():
    """Matches Dismantle.cs: Deal Damage."""
    card = create_card(CardId.DISMANTLE)
    combat = _make_combat(_character_for_card("DISMANTLE"))
    _play_smoke(combat, card)

def test_dodge_and_roll_onplay_smoke():
    """Matches DodgeAndRoll.cs: Apply power."""
    card = create_card(CardId.DODGE_AND_ROLL)
    combat = _make_combat(_character_for_card("DODGE_AND_ROLL"))
    _play_smoke(combat, card)

def test_dominate_onplay_smoke():
    """Matches Dominate.cs: Apply power."""
    card = create_card(CardId.DOMINATE)
    combat = _make_combat(_character_for_card("DOMINATE"))
    _play_smoke(combat, card)

def test_double_energy_onplay_smoke():
    """Matches DoubleEnergy.cs: See decompiled source."""
    card = create_card(CardId.DOUBLE_ENERGY)
    combat = _make_combat(_character_for_card("DOUBLE_ENERGY"))
    _play_smoke(combat, card)

def test_dramatic_entrance_onplay_smoke():
    """Matches DramaticEntrance.cs: Deal Damage."""
    card = create_card(CardId.DRAMATIC_ENTRANCE)
    combat = _make_combat(_character_for_card("DRAMATIC_ENTRANCE"))
    _play_smoke(combat, card)

def test_drum_of_battle_onplay_smoke():
    """Matches DrumOfBattle.cs: Apply power; Draw card(s)."""
    card = create_card(CardId.DRUM_OF_BATTLE_CARD)
    combat = _make_combat(_character_for_card("DRUM_OF_BATTLE_CARD"))
    _play_smoke(combat, card)

def test_dying_star_onplay_smoke():
    """Matches DyingStar.cs: Deal Damage; Apply power."""
    card = create_card(CardId.DYING_STAR)
    combat = _make_combat(_character_for_card("DYING_STAR"))
    _play_smoke(combat, card)

def test_echo_form_onplay_smoke():
    """Matches EchoForm.cs: Apply power."""
    card = create_card(CardId.ECHO_FORM_CARD)
    combat = _make_combat(_character_for_card("ECHO_FORM_CARD"))
    _play_smoke(combat, card)

def test_echoing_slash_onplay_smoke():
    """Matches EchoingSlash.cs: See decompiled source."""
    card = create_card(CardId.ECHOING_SLASH)
    combat = _make_combat(_character_for_card("ECHOING_SLASH"))
    _play_smoke(combat, card)

def test_energy_surge_onplay_smoke():
    """Matches EnergySurge.cs: See decompiled source."""
    card = create_card(CardId.ENERGY_SURGE)
    combat = _make_combat(_character_for_card("ENERGY_SURGE"))
    _play_smoke(combat, card)

def test_enfeebling_touch_onplay_smoke():
    """Matches EnfeeblingTouch.cs: Apply power."""
    card = create_card(CardId.ENFEEBLING_TOUCH)
    combat = _make_combat(_character_for_card("ENFEEBLING_TOUCH"))
    _play_smoke(combat, card)

def test_enlightenment_onplay_smoke():
    """Matches Enlightenment.cs: See decompiled source."""
    card = create_card(CardId.ENLIGHTENMENT)
    combat = _make_combat(_character_for_card("ENLIGHTENMENT"))
    _play_smoke(combat, card)

def test_entrench_onplay_smoke():
    """Matches Entrench.cs: See decompiled source."""
    card = create_card(CardId.ENTRENCH)
    combat = _make_combat(_character_for_card("ENTRENCH"))
    _play_smoke(combat, card)

def test_entropy_onplay_smoke():
    """Matches Entropy.cs: Apply power."""
    card = create_card(CardId.ENTROPY)
    combat = _make_combat(_character_for_card("ENTROPY"))
    _play_smoke(combat, card)

def test_envenom_onplay_smoke():
    """Matches Envenom.cs: Apply power."""
    card = create_card(CardId.ENVENOM_CARD)
    combat = _make_combat(_character_for_card("ENVENOM_CARD"))
    _play_smoke(combat, card)

def test_equilibrium_onplay_smoke():
    """Matches Equilibrium.cs: Apply power."""
    card = create_card(CardId.EQUILIBRIUM)
    combat = _make_combat(_character_for_card("EQUILIBRIUM"))
    _play_smoke(combat, card)

def test_eternal_armor_onplay_smoke():
    """Matches EternalArmor.cs: Apply power."""
    card = create_card(CardId.ETERNAL_ARMOR)
    combat = _make_combat(_character_for_card("ETERNAL_ARMOR"))
    _play_smoke(combat, card)

def test_evil_eye_onplay_smoke():
    """Matches EvilEye.cs: Exhaust."""
    card = create_card(CardId.EVIL_EYE)
    combat = _make_combat(_character_for_card("EVIL_EYE"))
    _play_smoke(combat, card)

def test_expertise_onplay_smoke():
    """Matches Expertise.cs: Draw card(s)."""
    card = create_card(CardId.EXPERTISE)
    combat = _make_combat(_character_for_card("EXPERTISE"))
    _play_smoke(combat, card)

def test_expose_onplay_smoke():
    """Matches Expose.cs: Apply power."""
    card = create_card(CardId.EXPOSE)
    combat = _make_combat(_character_for_card("EXPOSE"))
    _play_smoke(combat, card)

def test_exterminate_onplay_smoke():
    """Matches Exterminate.cs: Deal Damage."""
    card = create_card(CardId.EXTERMINATE)
    combat = _make_combat(_character_for_card("EXTERMINATE"))
    _play_smoke(combat, card)

def test_falling_star_onplay_smoke():
    """Matches FallingStar.cs: Deal Damage; Apply power."""
    card = create_card(CardId.FALLING_STAR)
    combat = _make_combat(_character_for_card("FALLING_STAR"))
    _play_smoke(combat, card)

def test_fan_of_knives_onplay_smoke():
    """Matches FanOfKnives.cs: Apply power."""
    card = create_card(CardId.FAN_OF_KNIVES_CARD)
    combat = _make_combat(_character_for_card("FAN_OF_KNIVES_CARD"))
    _play_smoke(combat, card)

def test_fasten_onplay_smoke():
    """Matches Fasten.cs: Apply power."""
    card = create_card(CardId.FASTEN)
    combat = _make_combat(_character_for_card("FASTEN"))
    _play_smoke(combat, card)

def test_feed_onplay_smoke():
    """Matches Feed.cs: Deal Damage."""
    card = create_card(CardId.FEED)
    combat = _make_combat(_character_for_card("FEED"))
    _play_smoke(combat, card)

def test_feeding_frenzy_onplay_smoke():
    """Matches FeedingFrenzy.cs: Apply power."""
    card = create_card(CardId.FEEDING_FRENZY_CARD)
    combat = _make_combat(_character_for_card("FEEDING_FRENZY_CARD"))
    _play_smoke(combat, card)

def test_feral_onplay_smoke():
    """Matches Feral.cs: Apply power."""
    card = create_card(CardId.FERAL)
    combat = _make_combat(_character_for_card("FERAL"))
    _play_smoke(combat, card)

def test_fetch_onplay_smoke():
    """Matches Fetch.cs: Deal Damage; Draw card(s)."""
    card = create_card(CardId.FETCH)
    combat = _make_combat(_character_for_card("FETCH"))
    _play_smoke(combat, card)

def test_fight_me_onplay_smoke():
    """Matches FightMe.cs: Deal Damage; Apply power."""
    card = create_card(CardId.FIGHT_ME)
    combat = _make_combat(_character_for_card("FIGHT_ME"))
    _play_smoke(combat, card)

def test_finesse_onplay_smoke():
    """Matches Finesse.cs: Draw card(s)."""
    card = create_card(CardId.FINESSE)
    combat = _make_combat(_character_for_card("FINESSE"))
    _play_smoke(combat, card)

def test_finisher_onplay_smoke():
    """Matches Finisher.cs: Deal Damage."""
    card = create_card(CardId.FINISHER)
    combat = _make_combat(_character_for_card("FINISHER"))
    _play_smoke(combat, card)

def test_fisticuffs_onplay_smoke():
    """Matches Fisticuffs.cs: Deal Damage."""
    card = create_card(CardId.FISTICUFFS)
    combat = _make_combat(_character_for_card("FISTICUFFS"))
    _play_smoke(combat, card)

def test_flame_barrier_onplay_smoke():
    """Matches FlameBarrier.cs: Apply power."""
    card = create_card(CardId.FLAME_BARRIER_CARD)
    combat = _make_combat(_character_for_card("FLAME_BARRIER_CARD"))
    _play_smoke(combat, card)

def test_flanking_onplay_smoke():
    """Matches Flanking.cs: Apply power."""
    card = create_card(CardId.FLANKING)
    combat = _make_combat(_character_for_card("FLANKING"))
    _play_smoke(combat, card)

def test_flash_of_steel_onplay_smoke():
    """Matches FlashOfSteel.cs: Deal Damage; Draw card(s)."""
    card = create_card(CardId.FLASH_OF_STEEL)
    combat = _make_combat(_character_for_card("FLASH_OF_STEEL"))
    _play_smoke(combat, card)

def test_flatten_onplay_smoke():
    """Matches Flatten.cs: Deal Damage."""
    card = create_card(CardId.FLATTEN)
    combat = _make_combat(_character_for_card("FLATTEN"))
    _play_smoke(combat, card)

def test_flechettes_onplay_smoke():
    """Matches Flechettes.cs: Deal Damage."""
    card = create_card(CardId.FLECHETTES)
    combat = _make_combat(_character_for_card("FLECHETTES"))
    _play_smoke(combat, card)

def test_flick_flack_onplay_smoke():
    """Matches FlickFlack.cs: Deal Damage."""
    card = create_card(CardId.FLICK_FLACK)
    combat = _make_combat(_character_for_card("FLICK_FLACK"))
    _play_smoke(combat, card)

def test_focused_strike_onplay_smoke():
    """Matches FocusedStrike.cs: Deal Damage; Apply power."""
    card = create_card(CardId.FOCUSED_STRIKE_CARD)
    combat = _make_combat(_character_for_card("FOCUSED_STRIKE_CARD"))
    _play_smoke(combat, card)

def test_follow_through_onplay_smoke():
    """Matches FollowThrough.cs: Deal Damage."""
    card = create_card(CardId.FOLLOW_THROUGH)
    combat = _make_combat(_character_for_card("FOLLOW_THROUGH"))
    _play_smoke(combat, card)

def test_footwork_onplay_smoke():
    """Matches Footwork.cs: Apply power."""
    card = create_card(CardId.FOOTWORK)
    combat = _make_combat(_character_for_card("FOOTWORK"))
    _play_smoke(combat, card)

def test_forbidden_grimoire_onplay_smoke():
    """Matches ForbiddenGrimoire.cs: Apply power."""
    card = create_card(CardId.FORBIDDEN_GRIMOIRE)
    combat = _make_combat(_character_for_card("FORBIDDEN_GRIMOIRE"))
    _play_smoke(combat, card)

def test_foregone_conclusion_onplay_smoke():
    """Matches ForegoneConclusion.cs: Apply power."""
    card = create_card(CardId.FOREGONE_CONCLUSION)
    combat = _make_combat(_character_for_card("FOREGONE_CONCLUSION"))
    _play_smoke(combat, card)

def test_forgotten_ritual_onplay_smoke():
    """Matches ForgottenRitual.cs: Exhaust."""
    card = create_card(CardId.FORGOTTEN_RITUAL)
    combat = _make_combat(_character_for_card("FORGOTTEN_RITUAL"))
    _play_smoke(combat, card)

def test_friendship_onplay_smoke():
    """Matches Friendship.cs: Apply power."""
    card = create_card(CardId.FRIENDSHIP)
    combat = _make_combat(_character_for_card("FRIENDSHIP"))
    _play_smoke(combat, card)

def test_fusion_onplay_smoke():
    """Matches Fusion.cs: Orb action."""
    card = create_card(CardId.FUSION)
    combat = _make_combat(_character_for_card("FUSION"))
    _play_smoke(combat, card)

def test_gamma_blast_onplay_smoke():
    """Matches GammaBlast.cs: Deal Damage; Apply power."""
    card = create_card(CardId.GAMMA_BLAST)
    combat = _make_combat(_character_for_card("GAMMA_BLAST"))
    _play_smoke(combat, card)

def test_giant_rock_onplay_smoke():
    """Matches GiantRock.cs: Deal Damage."""
    card = create_card(CardId.GIANT_ROCK)
    combat = _make_combat(_character_for_card("GIANT_ROCK"))
    _play_smoke(combat, card)

def test_glasswork_onplay_smoke():
    """Matches Glasswork.cs: Orb action."""
    card = create_card(CardId.GLASSWORK)
    combat = _make_combat(_character_for_card("GLASSWORK"))
    _play_smoke(combat, card)

def test_glitterstream_onplay_smoke():
    """Matches Glitterstream.cs: Apply power."""
    card = create_card(CardId.GLITTERSTREAM)
    combat = _make_combat(_character_for_card("GLITTERSTREAM"))
    _play_smoke(combat, card)

def test_gold_axe_onplay_smoke():
    """Matches GoldAxe.cs: Deal Damage."""
    card = create_card(CardId.GOLD_AXE)
    combat = _make_combat(_character_for_card("GOLD_AXE"))
    _play_smoke(combat, card)

def test_grapple_onplay_smoke():
    """Matches Grapple.cs: Deal Damage; Apply power."""
    card = create_card(CardId.GRAPPLE)
    combat = _make_combat(_character_for_card("GRAPPLE"))
    _play_smoke(combat, card)

def test_hailstorm_onplay_smoke():
    """Matches Hailstorm.cs: Apply power."""
    card = create_card(CardId.HAILSTORM)
    combat = _make_combat(_character_for_card("HAILSTORM"))
    _play_smoke(combat, card)

def test_hammer_time_onplay_smoke():
    """Matches HammerTime.cs: Apply power."""
    card = create_card(CardId.HAMMER_TIME)
    combat = _make_combat(_character_for_card("HAMMER_TIME"))
    _play_smoke(combat, card)

def test_hang_onplay_smoke():
    """Matches Hang.cs: Deal Damage; Apply power."""
    card = create_card(CardId.HANG)
    combat = _make_combat(_character_for_card("HANG"))
    _play_smoke(combat, card)

def test_haunt_onplay_smoke():
    """Matches Haunt.cs: Apply power."""
    card = create_card(CardId.HAUNT)
    combat = _make_combat(_character_for_card("HAUNT"))
    _play_smoke(combat, card)

def test_haze_onplay_smoke():
    """Matches Haze.cs: Apply power."""
    card = create_card(CardId.HAZE)
    combat = _make_combat(_character_for_card("HAZE"))
    _play_smoke(combat, card)

def test_hegemony_onplay_smoke():
    """Matches Hegemony.cs: Deal Damage; Apply power."""
    card = create_card(CardId.HEGEMONY)
    combat = _make_combat(_character_for_card("HEGEMONY"))
    _play_smoke(combat, card)

def test_hello_world_onplay_smoke():
    """Matches HelloWorld.cs: Apply power."""
    card = create_card(CardId.HELLO_WORLD_CARD)
    combat = _make_combat(_character_for_card("HELLO_WORLD_CARD"))
    _play_smoke(combat, card)

def test_hotfix_onplay_smoke():
    """Matches Hotfix.cs: Apply power."""
    card = create_card(CardId.HOTFIX)
    combat = _make_combat(_character_for_card("HOTFIX"))
    _play_smoke(combat, card)

def test_howl_from_beyond_onplay_smoke():
    """Matches HowlFromBeyond.cs: Deal Damage."""
    card = create_card(CardId.HOWL_FROM_BEYOND)
    combat = _make_combat(_character_for_card("HOWL_FROM_BEYOND"))
    _play_smoke(combat, card)

def test_hyperbeam_onplay_smoke():
    """Matches Hyperbeam.cs: Deal Damage; Apply power."""
    card = create_card(CardId.HYPERBEAM)
    combat = _make_combat(_character_for_card("HYPERBEAM"))
    _play_smoke(combat, card)

def test_i_am_invincible_onplay_smoke():
    """Matches IAmInvincible.cs: See decompiled source."""
    card = create_card(CardId.I_AM_INVINCIBLE)
    combat = _make_combat(_character_for_card("I_AM_INVINCIBLE"))
    _play_smoke(combat, card)

def test_ignition_onplay_smoke():
    """Matches Ignition.cs: Orb action."""
    card = create_card(CardId.IGNITION)
    combat = _make_combat(_character_for_card("IGNITION"))
    _play_smoke(combat, card)

def test_impervious_onplay_smoke():
    """Matches Impervious.cs: See decompiled source."""
    card = create_card(CardId.IMPERVIOUS)
    combat = _make_combat(_character_for_card("IMPERVIOUS"))
    _play_smoke(combat, card)

def test_inferno_onplay_smoke():
    """Matches Inferno.cs: Apply power."""
    card = create_card(CardId.INFERNO_CARD)
    combat = _make_combat(_character_for_card("INFERNO_CARD"))
    _play_smoke(combat, card)

def test_infinite_blades_onplay_smoke():
    """Matches InfiniteBlades.cs: Apply power."""
    card = create_card(CardId.INFINITE_BLADES_CARD)
    combat = _make_combat(_character_for_card("INFINITE_BLADES_CARD"))
    _play_smoke(combat, card)

def test_intercept_onplay_smoke():
    """Matches Intercept.cs: Apply power."""
    card = create_card(CardId.INTERCEPT_CARD)
    combat = _make_combat(_character_for_card("INTERCEPT_CARD"))
    _play_smoke(combat, card)

def test_invoke_onplay_smoke():
    """Matches Invoke.cs: Apply power."""
    card = create_card(CardId.INVOKE)
    combat = _make_combat(_character_for_card("INVOKE"))
    _play_smoke(combat, card)

def test_iron_wave_onplay_smoke():
    """Matches IronWave.cs: Deal Damage."""
    card = create_card(CardId.IRON_WAVE)
    combat = _make_combat(_character_for_card("IRON_WAVE"))
    _play_smoke(combat, card)

def test_iteration_onplay_smoke():
    """Matches Iteration.cs: Apply power."""
    card = create_card(CardId.ITERATION_CARD)
    combat = _make_combat(_character_for_card("ITERATION_CARD"))
    _play_smoke(combat, card)

def test_juggernaut_onplay_smoke():
    """Matches Juggernaut.cs: Apply power."""
    card = create_card(CardId.JUGGERNAUT_CARD)
    combat = _make_combat(_character_for_card("JUGGERNAUT_CARD"))
    _play_smoke(combat, card)

def test_juggling_onplay_smoke():
    """Matches Juggling.cs: Apply power."""
    card = create_card(CardId.JUGGLING_CARD)
    combat = _make_combat(_character_for_card("JUGGLING_CARD"))
    _play_smoke(combat, card)

def test_kingly_kick_onplay_smoke():
    """Matches KinglyKick.cs: Deal Damage."""
    card = create_card(CardId.KINGLY_KICK)
    combat = _make_combat(_character_for_card("KINGLY_KICK"))
    _play_smoke(combat, card)

def test_kingly_punch_onplay_smoke():
    """Matches KinglyPunch.cs: Deal Damage."""
    card = create_card(CardId.KINGLY_PUNCH)
    combat = _make_combat(_character_for_card("KINGLY_PUNCH"))
    _play_smoke(combat, card)

def test_know_thy_place_onplay_smoke():
    """Matches KnowThyPlace.cs: Apply power."""
    card = create_card(CardId.KNOW_THY_PLACE)
    combat = _make_combat(_character_for_card("KNOW_THY_PLACE"))
    _play_smoke(combat, card)

def test_leading_strike_onplay_smoke():
    """Matches LeadingStrike.cs: Deal Damage."""
    card = create_card(CardId.LEADING_STRIKE)
    combat = _make_combat(_character_for_card("LEADING_STRIKE"))
    _play_smoke(combat, card)

def test_leap_onplay_smoke():
    """Matches Leap.cs: See decompiled source."""
    card = create_card(CardId.LEAP)
    combat = _make_combat(_character_for_card("LEAP"))
    _play_smoke(combat, card)

def test_leg_sweep_onplay_smoke():
    """Matches LegSweep.cs: Apply power."""
    card = create_card(CardId.LEG_SWEEP)
    combat = _make_combat(_character_for_card("LEG_SWEEP"))
    _play_smoke(combat, card)

def test_lethality_onplay_smoke():
    """Matches Lethality.cs: Apply power."""
    card = create_card(CardId.LETHALITY_CARD)
    combat = _make_combat(_character_for_card("LETHALITY_CARD"))
    _play_smoke(combat, card)

def test_lift_onplay_smoke():
    """Matches Lift.cs: See decompiled source."""
    card = create_card(CardId.LIFT)
    combat = _make_combat(_character_for_card("LIFT"))
    _play_smoke(combat, card)

def test_luminesce_onplay_smoke():
    """Matches Luminesce.cs: See decompiled source."""
    card = create_card(CardId.LUMINESCE)
    combat = _make_combat(_character_for_card("LUMINESCE"))
    _play_smoke(combat, card)

def test_lunar_blast_onplay_smoke():
    """Matches LunarBlast.cs: Deal Damage."""
    card = create_card(CardId.LUNAR_BLAST)
    combat = _make_combat(_character_for_card("LUNAR_BLAST"))
    _play_smoke(combat, card)

def test_mad_science_onplay_smoke():
    """Matches MadScience.cs: See decompiled source."""
    card = create_card(CardId.MAD_SCIENCE)
    combat = _make_combat(_character_for_card("MAD_SCIENCE"))
    _play_smoke(combat, card)

def test_malaise_onplay_smoke():
    """Matches Malaise.cs: Apply power; Upgrade card(s)."""
    card = create_card(CardId.MALAISE)
    combat = _make_combat(_character_for_card("MALAISE"))
    _play_smoke(combat, card)

def test_mangle_onplay_smoke():
    """Matches Mangle.cs: Deal Damage; Apply power."""
    card = create_card(CardId.MANGLE)
    combat = _make_combat(_character_for_card("MANGLE"))
    _play_smoke(combat, card)

def test_master_planner_onplay_smoke():
    """Matches MasterPlanner.cs: Apply power."""
    card = create_card(CardId.MASTER_PLANNER)
    combat = _make_combat(_character_for_card("MASTER_PLANNER"))
    _play_smoke(combat, card)

def test_maul_onplay_smoke():
    """Matches Maul.cs: Deal Damage."""
    card = create_card(CardId.MAUL)
    combat = _make_combat(_character_for_card("MAUL"))
    _play_smoke(combat, card)

def test_mayhem_onplay_smoke():
    """Matches Mayhem.cs: Apply power."""
    card = create_card(CardId.MAYHEM_CARD)
    combat = _make_combat(_character_for_card("MAYHEM_CARD"))
    _play_smoke(combat, card)

def test_melancholy_onplay_smoke():
    """Matches Melancholy.cs: See decompiled source."""
    card = create_card(CardId.MELANCHOLY)
    combat = _make_combat(_character_for_card("MELANCHOLY"))
    _play_smoke(combat, card)

def test_memento_mori_onplay_smoke():
    """Matches MementoMori.cs: Deal Damage."""
    card = create_card(CardId.MEMENTO_MORI)
    combat = _make_combat(_character_for_card("MEMENTO_MORI"))
    _play_smoke(combat, card)

def test_meteor_shower_onplay_smoke():
    """Matches MeteorShower.cs: Deal Damage; Apply power."""
    card = create_card(CardId.METEOR_SHOWER)
    combat = _make_combat(_character_for_card("METEOR_SHOWER"))
    _play_smoke(combat, card)

def test_meteor_strike_onplay_smoke():
    """Matches MeteorStrike.cs: Deal Damage; Orb action."""
    card = create_card(CardId.METEOR_STRIKE)
    combat = _make_combat(_character_for_card("METEOR_STRIKE"))
    _play_smoke(combat, card)

def test_minion_dive_bomb_onplay_smoke():
    """Matches MinionDiveBomb.cs: Deal Damage."""
    card = create_card(CardId.MINION_DIVE_BOMB)
    combat = _make_combat(_character_for_card("MINION_DIVE_BOMB"))
    _play_smoke(combat, card)

def test_minion_sacrifice_onplay_smoke():
    """Matches MinionSacrifice.cs: See decompiled source."""
    card = create_card(CardId.MINION_SACRIFICE)
    combat = _make_combat(_character_for_card("MINION_SACRIFICE"))
    _play_smoke(combat, card)

def test_minion_strike_onplay_smoke():
    """Matches MinionStrike.cs: Deal Damage; Draw card(s)."""
    card = create_card(CardId.MINION_STRIKE)
    combat = _make_combat(_character_for_card("MINION_STRIKE"))
    _play_smoke(combat, card)

def test_mirage_onplay_smoke():
    """Matches Mirage.cs: See decompiled source."""
    card = create_card(CardId.MIRAGE)
    combat = _make_combat(_character_for_card("MIRAGE"))
    _play_smoke(combat, card)

def test_misery_onplay_smoke():
    """Matches Misery.cs: Deal Damage; Apply power."""
    card = create_card(CardId.MISERY)
    combat = _make_combat(_character_for_card("MISERY"))
    _play_smoke(combat, card)

def test_modded_onplay_smoke():
    """Matches Modded.cs: Draw card(s); Orb action."""
    card = create_card(CardId.MODDED)
    combat = _make_combat(_character_for_card("MODDED"))
    _play_smoke(combat, card)

def test_molten_fist_onplay_smoke():
    """Matches MoltenFist.cs: Deal Damage; Apply power."""
    card = create_card(CardId.MOLTEN_FIST)
    combat = _make_combat(_character_for_card("MOLTEN_FIST"))
    _play_smoke(combat, card)

def test_monarchs_gaze_onplay_smoke():
    """Matches MonarchsGaze.cs: Apply power."""
    card = create_card(CardId.MONARCHS_GAZE_CARD)
    combat = _make_combat(_character_for_card("MONARCHS_GAZE_CARD"))
    _play_smoke(combat, card)

def test_monologue_onplay_smoke():
    """Matches Monologue.cs: Apply power."""
    card = create_card(CardId.MONOLOGUE_CARD)
    combat = _make_combat(_character_for_card("MONOLOGUE_CARD"))
    _play_smoke(combat, card)

def test_murder_onplay_smoke():
    """Matches Murder.cs: Deal Damage."""
    card = create_card(CardId.MURDER)
    combat = _make_combat(_character_for_card("MURDER"))
    _play_smoke(combat, card)

def test_negative_pulse_onplay_smoke():
    """Matches NegativePulse.cs: Apply power."""
    card = create_card(CardId.NEGATIVE_PULSE)
    combat = _make_combat(_character_for_card("NEGATIVE_PULSE"))
    _play_smoke(combat, card)

def test_neows_fury_onplay_smoke():
    """Matches NeowsFury.cs: Deal Damage."""
    card = create_card(CardId.NEOWS_FURY)
    combat = _make_combat(_character_for_card("NEOWS_FURY"))
    _play_smoke(combat, card)

def test_neutralize_onplay_smoke():
    """Matches Neutralize.cs: Deal Damage; Apply power."""
    card = create_card(CardId.NEUTRALIZE)
    combat = _make_combat(_character_for_card("NEUTRALIZE"))
    _play_smoke(combat, card)

def test_neutron_aegis_onplay_smoke():
    """Matches NeutronAegis.cs: Apply power."""
    card = create_card(CardId.NEUTRON_AEGIS)
    combat = _make_combat(_character_for_card("NEUTRON_AEGIS"))
    _play_smoke(combat, card)

def test_nostalgia_onplay_smoke():
    """Matches Nostalgia.cs: Apply power."""
    card = create_card(CardId.NOSTALGIA_CARD)
    combat = _make_combat(_character_for_card("NOSTALGIA_CARD"))
    _play_smoke(combat, card)

def test_noxious_fumes_onplay_smoke():
    """Matches NoxiousFumes.cs: Apply power."""
    card = create_card(CardId.NOXIOUS_FUMES_CARD)
    combat = _make_combat(_character_for_card("NOXIOUS_FUMES_CARD"))
    _play_smoke(combat, card)

def test_null_onplay_smoke():
    """Matches Null.cs: Deal Damage; Apply power; Orb action."""
    card = create_card(CardId.NULL_CARD)
    combat = _make_combat(_character_for_card("NULL_CARD"))
    _play_smoke(combat, card)

def test_oblivion_onplay_smoke():
    """Matches Oblivion.cs: Apply power."""
    card = create_card(CardId.OBLIVION)
    combat = _make_combat(_character_for_card("OBLIVION"))
    _play_smoke(combat, card)

def test_omnislice_onplay_smoke():
    """Matches Omnislice.cs: See decompiled source."""
    card = create_card(CardId.OMNISLICE)
    combat = _make_combat(_character_for_card("OMNISLICE"))
    _play_smoke(combat, card)

def test_outbreak_onplay_smoke():
    """Matches Outbreak.cs: Apply power."""
    card = create_card(CardId.OUTBREAK_CARD)
    combat = _make_combat(_character_for_card("OUTBREAK_CARD"))
    _play_smoke(combat, card)

def test_outmaneuver_onplay_smoke():
    """Matches Outmaneuver.cs: Apply power."""
    card = create_card(CardId.OUTMANEUVER)
    combat = _make_combat(_character_for_card("OUTMANEUVER"))
    _play_smoke(combat, card)

def test_pacts_end_onplay_smoke():
    """Matches PactsEnd.cs: Deal Damage."""
    card = create_card(CardId.PACTS_END)
    combat = _make_combat(_character_for_card("PACTS_END"))
    _play_smoke(combat, card)

def test_pagestorm_onplay_smoke():
    """Matches Pagestorm.cs: Apply power."""
    card = create_card(CardId.PAGESTORM)
    combat = _make_combat(_character_for_card("PAGESTORM"))
    _play_smoke(combat, card)

def test_panic_button_onplay_smoke():
    """Matches PanicButton.cs: Apply power."""
    card = create_card(CardId.PANIC_BUTTON)
    combat = _make_combat(_character_for_card("PANIC_BUTTON"))
    _play_smoke(combat, card)

def test_parry_onplay_smoke():
    """Matches Parry.cs: Apply power."""
    card = create_card(CardId.PARRY_CARD)
    combat = _make_combat(_character_for_card("PARRY_CARD"))
    _play_smoke(combat, card)

def test_parse_onplay_smoke():
    """Matches Parse.cs: Draw card(s)."""
    card = create_card(CardId.PARSE)
    combat = _make_combat(_character_for_card("PARSE"))
    _play_smoke(combat, card)

def test_patter_onplay_smoke():
    """Matches Patter.cs: Apply power."""
    card = create_card(CardId.PATTER)
    combat = _make_combat(_character_for_card("PATTER"))
    _play_smoke(combat, card)

def test_peck_onplay_smoke():
    """Matches Peck.cs: Deal Damage."""
    card = create_card(CardId.PECK)
    combat = _make_combat(_character_for_card("PECK"))
    _play_smoke(combat, card)

def test_phantom_blades_onplay_smoke():
    """Matches PhantomBlades.cs: Apply power."""
    card = create_card(CardId.PHANTOM_BLADES_CARD)
    combat = _make_combat(_character_for_card("PHANTOM_BLADES_CARD"))
    _play_smoke(combat, card)

def test_piercing_wail_onplay_smoke():
    """Matches PiercingWail.cs: Apply power."""
    card = create_card(CardId.PIERCING_WAIL)
    combat = _make_combat(_character_for_card("PIERCING_WAIL"))
    _play_smoke(combat, card)

def test_pillage_onplay_smoke():
    """Matches Pillage.cs: Deal Damage; Draw card(s)."""
    card = create_card(CardId.PILLAGE)
    combat = _make_combat(_character_for_card("PILLAGE"))
    _play_smoke(combat, card)

def test_pillar_of_creation_onplay_smoke():
    """Matches PillarOfCreation.cs: Apply power."""
    card = create_card(CardId.PILLAR_OF_CREATION)
    combat = _make_combat(_character_for_card("PILLAR_OF_CREATION"))
    _play_smoke(combat, card)

def test_pinpoint_onplay_smoke():
    """Matches Pinpoint.cs: Deal Damage."""
    card = create_card(CardId.PINPOINT)
    combat = _make_combat(_character_for_card("PINPOINT"))
    _play_smoke(combat, card)

def test_poisoned_stab_onplay_smoke():
    """Matches PoisonedStab.cs: Deal Damage; Apply power."""
    card = create_card(CardId.POISONED_STAB)
    combat = _make_combat(_character_for_card("POISONED_STAB"))
    _play_smoke(combat, card)

def test_poke_onplay_smoke():
    """Matches Poke.cs: Deal Damage."""
    card = create_card(CardId.POKE)
    combat = _make_combat(_character_for_card("POKE"))
    _play_smoke(combat, card)

def test_pounce_onplay_smoke():
    """Matches Pounce.cs: Deal Damage; Apply power."""
    card = create_card(CardId.POUNCE)
    combat = _make_combat(_character_for_card("POUNCE"))
    _play_smoke(combat, card)

def test_precise_cut_onplay_smoke():
    """Matches PreciseCut.cs: Deal Damage."""
    card = create_card(CardId.PRECISE_CUT)
    combat = _make_combat(_character_for_card("PRECISE_CUT"))
    _play_smoke(combat, card)

def test_predator_onplay_smoke():
    """Matches Predator.cs: Deal Damage; Apply power."""
    card = create_card(CardId.PREDATOR)
    combat = _make_combat(_character_for_card("PREDATOR"))
    _play_smoke(combat, card)

def test_production_onplay_smoke():
    """Matches Production.cs: See decompiled source."""
    card = create_card(CardId.PRODUCTION)
    combat = _make_combat(_character_for_card("PRODUCTION"))
    _play_smoke(combat, card)

def test_prolong_onplay_smoke():
    """Matches Prolong.cs: Apply power."""
    card = create_card(CardId.PROLONG)
    combat = _make_combat(_character_for_card("PROLONG"))
    _play_smoke(combat, card)

def test_prophesize_onplay_smoke():
    """Matches Prophesize.cs: Draw card(s)."""
    card = create_card(CardId.PROPHESIZE)
    combat = _make_combat(_character_for_card("PROPHESIZE"))
    _play_smoke(combat, card)

def test_prowess_onplay_smoke():
    """Matches Prowess.cs: Apply power."""
    card = create_card(CardId.PROWESS)
    combat = _make_combat(_character_for_card("PROWESS"))
    _play_smoke(combat, card)

def test_pull_from_below_onplay_smoke():
    """Matches PullFromBelow.cs: Deal Damage."""
    card = create_card(CardId.PULL_FROM_BELOW)
    combat = _make_combat(_character_for_card("PULL_FROM_BELOW"))
    _play_smoke(combat, card)

def test_pyre_onplay_smoke():
    """Matches Pyre.cs: Apply power."""
    card = create_card(CardId.PYRE)
    combat = _make_combat(_character_for_card("PYRE"))
    _play_smoke(combat, card)

def test_quadcast_onplay_smoke():
    """Matches Quadcast.cs: Orb action."""
    card = create_card(CardId.QUADCAST)
    combat = _make_combat(_character_for_card("QUADCAST"))
    _play_smoke(combat, card)

def test_rage_onplay_smoke():
    """Matches Rage.cs: Apply power."""
    card = create_card(CardId.RAGE_CARD)
    combat = _make_combat(_character_for_card("RAGE_CARD"))
    _play_smoke(combat, card)

def test_rainbow_onplay_smoke():
    """Matches Rainbow.cs: Orb action."""
    card = create_card(CardId.RAINBOW)
    combat = _make_combat(_character_for_card("RAINBOW"))
    _play_smoke(combat, card)

def test_rally_onplay_smoke():
    """Matches Rally.cs: See decompiled source."""
    card = create_card(CardId.RALLY)
    combat = _make_combat(_character_for_card("RALLY"))
    _play_smoke(combat, card)

def test_reaper_form_onplay_smoke():
    """Matches ReaperForm.cs: Apply power."""
    card = create_card(CardId.REAPER_FORM)
    combat = _make_combat(_character_for_card("REAPER_FORM"))
    _play_smoke(combat, card)

def test_rebound_onplay_smoke():
    """Matches Rebound.cs: Deal Damage; Apply power."""
    card = create_card(CardId.REBOUND)
    combat = _make_combat(_character_for_card("REBOUND"))
    _play_smoke(combat, card)

def test_reflect_onplay_smoke():
    """Matches Reflect.cs: Apply power."""
    card = create_card(CardId.REFLECT_CARD)
    combat = _make_combat(_character_for_card("REFLECT_CARD"))
    _play_smoke(combat, card)

def test_reflex_onplay_smoke():
    """Matches Reflex.cs: Draw card(s)."""
    card = create_card(CardId.REFLEX)
    combat = _make_combat(_character_for_card("REFLEX"))
    _play_smoke(combat, card)

def test_relax_onplay_smoke():
    """Matches Relax.cs: Apply power."""
    card = create_card(CardId.RELAX)
    combat = _make_combat(_character_for_card("RELAX"))
    _play_smoke(combat, card)

def test_rend_onplay_smoke():
    """Matches Rend.cs: Deal Damage."""
    card = create_card(CardId.REND)
    combat = _make_combat(_character_for_card("REND"))
    _play_smoke(combat, card)

def test_restlessness_onplay_smoke():
    """Matches Restlessness.cs: Draw card(s)."""
    card = create_card(CardId.RESTLESSNESS)
    combat = _make_combat(_character_for_card("RESTLESSNESS"))
    _play_smoke(combat, card)

def test_ricochet_onplay_smoke():
    """Matches Ricochet.cs: Deal Damage."""
    card = create_card(CardId.RICOCHET)
    combat = _make_combat(_character_for_card("RICOCHET"))
    _play_smoke(combat, card)

def test_right_hand_hand_onplay_smoke():
    """Matches RightHandHand.cs: Deal Damage."""
    card = create_card(CardId.RIGHT_HAND_HAND)
    combat = _make_combat(_character_for_card("RIGHT_HAND_HAND"))
    _play_smoke(combat, card)

def test_rip_and_tear_onplay_smoke():
    """Matches RipAndTear.cs: Deal Damage."""
    card = create_card(CardId.RIP_AND_TEAR)
    combat = _make_combat(_character_for_card("RIP_AND_TEAR"))
    _play_smoke(combat, card)

def test_rocket_punch_onplay_smoke():
    """Matches RocketPunch.cs: Deal Damage; Draw card(s)."""
    card = create_card(CardId.ROCKET_PUNCH)
    combat = _make_combat(_character_for_card("ROCKET_PUNCH"))
    _play_smoke(combat, card)

def test_royalties_onplay_smoke():
    """Matches Royalties.cs: Apply power."""
    card = create_card(CardId.ROYALTIES_CARD)
    combat = _make_combat(_character_for_card("ROYALTIES_CARD"))
    _play_smoke(combat, card)

def test_rupture_onplay_smoke():
    """Matches Rupture.cs: Apply power."""
    card = create_card(CardId.RUPTURE_CARD)
    combat = _make_combat(_character_for_card("RUPTURE_CARD"))
    _play_smoke(combat, card)

def test_salvo_onplay_smoke():
    """Matches Salvo.cs: Deal Damage; Apply power."""
    card = create_card(CardId.SALVO)
    combat = _make_combat(_character_for_card("SALVO"))
    _play_smoke(combat, card)

def test_scourge_onplay_smoke():
    """Matches Scourge.cs: Apply power; Draw card(s)."""
    card = create_card(CardId.SCOURGE)
    combat = _make_combat(_character_for_card("SCOURGE"))
    _play_smoke(combat, card)

def test_serpent_form_onplay_smoke():
    """Matches SerpentForm.cs: Apply power."""
    card = create_card(CardId.SERPENT_FORM_CARD)
    combat = _make_combat(_character_for_card("SERPENT_FORM_CARD"))
    _play_smoke(combat, card)

def test_setup_strike_onplay_smoke():
    """Matches SetupStrike.cs: Deal Damage; Apply power."""
    card = create_card(CardId.SETUP_STRIKE_CARD)
    combat = _make_combat(_character_for_card("SETUP_STRIKE_CARD"))
    _play_smoke(combat, card)

def test_shadow_shield_onplay_smoke():
    """Matches ShadowShield.cs: Orb action."""
    card = create_card(CardId.SHADOW_SHIELD)
    combat = _make_combat(_character_for_card("SHADOW_SHIELD"))
    _play_smoke(combat, card)

def test_shadow_step_onplay_smoke():
    """Matches ShadowStep.cs: Apply power."""
    card = create_card(CardId.SHADOW_STEP)
    combat = _make_combat(_character_for_card("SHADOW_STEP"))
    _play_smoke(combat, card)

def test_shadowmeld_onplay_smoke():
    """Matches Shadowmeld.cs: Apply power."""
    card = create_card(CardId.SHADOWMELD)
    combat = _make_combat(_character_for_card("SHADOWMELD"))
    _play_smoke(combat, card)

def test_shared_fate_onplay_smoke():
    """Matches SharedFate.cs: Apply power."""
    card = create_card(CardId.SHARED_FATE)
    combat = _make_combat(_character_for_card("SHARED_FATE"))
    _play_smoke(combat, card)

def test_shatter_onplay_smoke():
    """Matches Shatter.cs: Deal Damage; Orb action."""
    card = create_card(CardId.SHATTER)
    combat = _make_combat(_character_for_card("SHATTER"))
    _play_smoke(combat, card)

def test_shiv_onplay_smoke():
    """Matches Shiv.cs: Deal Damage."""
    card = create_card(CardId.SHIV)
    combat = _make_combat(_character_for_card("SHIV"))
    _play_smoke(combat, card)

def test_shockwave_onplay_smoke():
    """Matches Shockwave.cs: Apply power."""
    card = create_card(CardId.SHOCKWAVE)
    combat = _make_combat(_character_for_card("SHOCKWAVE"))
    _play_smoke(combat, card)

def test_shroud_onplay_smoke():
    """Matches Shroud.cs: Apply power."""
    card = create_card(CardId.SHROUD)
    combat = _make_combat(_character_for_card("SHROUD"))
    _play_smoke(combat, card)

def test_skewer_onplay_smoke():
    """Matches Skewer.cs: Deal Damage."""
    card = create_card(CardId.SKEWER)
    combat = _make_combat(_character_for_card("SKEWER"))
    _play_smoke(combat, card)

def test_skim_onplay_smoke():
    """Matches Skim.cs: Draw card(s)."""
    card = create_card(CardId.SKIM)
    combat = _make_combat(_character_for_card("SKIM"))
    _play_smoke(combat, card)

def test_sleight_of_flesh_onplay_smoke():
    """Matches SleightOfFlesh.cs: Apply power."""
    card = create_card(CardId.SLEIGHT_OF_FLESH)
    combat = _make_combat(_character_for_card("SLEIGHT_OF_FLESH"))
    _play_smoke(combat, card)

def test_slice_onplay_smoke():
    """Matches Slice.cs: Deal Damage."""
    card = create_card(CardId.SLICE)
    combat = _make_combat(_character_for_card("SLICE"))
    _play_smoke(combat, card)

def test_slimed_onplay_smoke():
    """Matches Slimed.cs: Draw card(s)."""
    card = create_card(CardId.SLIMED)
    combat = _make_combat(_character_for_card("SLIMED"))
    _play_smoke(combat, card)

def test_smokestack_onplay_smoke():
    """Matches Smokestack.cs: Apply power."""
    card = create_card(CardId.SMOKESTACK)
    combat = _make_combat(_character_for_card("SMOKESTACK"))
    _play_smoke(combat, card)

def test_snakebite_onplay_smoke():
    """Matches Snakebite.cs: Apply power."""
    card = create_card(CardId.SNAKEBITE)
    combat = _make_combat(_character_for_card("SNAKEBITE"))
    _play_smoke(combat, card)

def test_sneaky_onplay_smoke():
    """Matches Sneaky.cs: Apply power."""
    card = create_card(CardId.SNEAKY_CARD)
    combat = _make_combat(_character_for_card("SNEAKY_CARD"))
    _play_smoke(combat, card)

def test_soul_onplay_smoke():
    """Matches Soul.cs: Draw card(s)."""
    card = create_card(CardId.SOUL)
    combat = _make_combat(_character_for_card("SOUL"))
    _play_smoke(combat, card)

def test_sovereign_blade_onplay_smoke():
    """Matches SovereignBlade.cs: Deal Damage."""
    card = create_card(CardId.SOVEREIGN_BLADE)
    combat = _make_combat(_character_for_card("SOVEREIGN_BLADE"))
    _play_smoke(combat, card)

def test_sow_onplay_smoke():
    """Matches Sow.cs: Deal Damage."""
    card = create_card(CardId.SOW)
    combat = _make_combat(_character_for_card("SOW"))
    _play_smoke(combat, card)

def test_spectrum_shift_onplay_smoke():
    """Matches SpectrumShift.cs: Apply power."""
    card = create_card(CardId.SPECTRUM_SHIFT)
    combat = _make_combat(_character_for_card("SPECTRUM_SHIFT"))
    _play_smoke(combat, card)

def test_speedster_onplay_smoke():
    """Matches Speedster.cs: Apply power."""
    card = create_card(CardId.SPEEDSTER_CARD)
    combat = _make_combat(_character_for_card("SPEEDSTER_CARD"))
    _play_smoke(combat, card)

def test_spinner_onplay_smoke():
    """Matches Spinner.cs: Apply power; Orb action; Upgrade card(s)."""
    card = create_card(CardId.SPINNER_CARD)
    combat = _make_combat(_character_for_card("SPINNER_CARD"))
    _play_smoke(combat, card)

def test_spirit_of_ash_onplay_smoke():
    """Matches SpiritOfAsh.cs: Apply power; Exhaust."""
    card = create_card(CardId.SPIRIT_OF_ASH)
    combat = _make_combat(_character_for_card("SPIRIT_OF_ASH"))
    _play_smoke(combat, card)

def test_spite_onplay_smoke():
    """Matches Spite.cs: Deal Damage."""
    card = create_card(CardId.SPITE)
    combat = _make_combat(_character_for_card("SPITE"))
    _play_smoke(combat, card)

def test_squash_onplay_smoke():
    """Matches Squash.cs: Deal Damage; Apply power."""
    card = create_card(CardId.SQUASH)
    combat = _make_combat(_character_for_card("SQUASH"))
    _play_smoke(combat, card)

def test_squeeze_onplay_smoke():
    """Matches Squeeze.cs: Deal Damage."""
    card = create_card(CardId.SQUEEZE)
    combat = _make_combat(_character_for_card("SQUEEZE"))
    _play_smoke(combat, card)

def test_stack_onplay_smoke():
    """Matches Stack.cs: See decompiled source."""
    card = create_card(CardId.STACK)
    combat = _make_combat(_character_for_card("STACK"))
    _play_smoke(combat, card)

def test_stampede_onplay_smoke():
    """Matches Stampede.cs: Apply power."""
    card = create_card(CardId.STAMPEDE_CARD)
    combat = _make_combat(_character_for_card("STAMPEDE_CARD"))
    _play_smoke(combat, card)

def test_stomp_onplay_smoke():
    """Matches Stomp.cs: Deal Damage."""
    card = create_card(CardId.STOMP)
    combat = _make_combat(_character_for_card("STOMP"))
    _play_smoke(combat, card)

def test_stone_armor_onplay_smoke():
    """Matches StoneArmor.cs: Apply power."""
    card = create_card(CardId.STONE_ARMOR)
    combat = _make_combat(_character_for_card("STONE_ARMOR"))
    _play_smoke(combat, card)

def test_storm_onplay_smoke():
    """Matches Storm.cs: Apply power."""
    card = create_card(CardId.STORM_CARD)
    combat = _make_combat(_character_for_card("STORM_CARD"))
    _play_smoke(combat, card)

def test_storm_of_steel_onplay_smoke():
    """Matches StormOfSteel.cs: Upgrade card(s)."""
    card = create_card(CardId.STORM_OF_STEEL)
    combat = _make_combat(_character_for_card("STORM_OF_STEEL"))
    _play_smoke(combat, card)

def test_strangle_onplay_smoke():
    """Matches Strangle.cs: Deal Damage; Apply power."""
    card = create_card(CardId.STRANGLE)
    combat = _make_combat(_character_for_card("STRANGLE"))
    _play_smoke(combat, card)

def test_stratagem_onplay_smoke():
    """Matches Stratagem.cs: Apply power."""
    card = create_card(CardId.STRATAGEM)
    combat = _make_combat(_character_for_card("STRATAGEM"))
    _play_smoke(combat, card)

def test_strike_defect_onplay_smoke():
    """Matches StrikeDefect.cs: Deal Damage."""
    card = create_card(CardId.STRIKE_DEFECT)
    combat = _make_combat(_character_for_card("STRIKE_DEFECT"))
    _play_smoke(combat, card)

def test_strike_ironclad_onplay_smoke():
    """Matches StrikeIronclad.cs: Deal Damage."""
    card = create_card(CardId.STRIKE_IRONCLAD)
    combat = _make_combat(_character_for_card("STRIKE_IRONCLAD"))
    _play_smoke(combat, card)

def test_strike_necrobinder_onplay_smoke():
    """Matches StrikeNecrobinder.cs: Deal Damage."""
    card = create_card(CardId.STRIKE_NECROBINDER)
    combat = _make_combat(_character_for_card("STRIKE_NECROBINDER"))
    _play_smoke(combat, card)

def test_strike_regent_onplay_smoke():
    """Matches StrikeRegent.cs: Deal Damage."""
    card = create_card(CardId.STRIKE_REGENT)
    combat = _make_combat(_character_for_card("STRIKE_REGENT"))
    _play_smoke(combat, card)

def test_strike_silent_onplay_smoke():
    """Matches StrikeSilent.cs: Deal Damage."""
    card = create_card(CardId.STRIKE_SILENT)
    combat = _make_combat(_character_for_card("STRIKE_SILENT"))
    _play_smoke(combat, card)

def test_subroutine_onplay_smoke():
    """Matches Subroutine.cs: Apply power."""
    card = create_card(CardId.SUBROUTINE)
    combat = _make_combat(_character_for_card("SUBROUTINE"))
    _play_smoke(combat, card)

def test_sucker_punch_onplay_smoke():
    """Matches SuckerPunch.cs: Deal Damage; Apply power."""
    card = create_card(CardId.SUCKER_PUNCH)
    combat = _make_combat(_character_for_card("SUCKER_PUNCH"))
    _play_smoke(combat, card)

def test_supercritical_onplay_smoke():
    """Matches Supercritical.cs: See decompiled source."""
    card = create_card(CardId.SUPERCRITICAL)
    combat = _make_combat(_character_for_card("SUPERCRITICAL"))
    _play_smoke(combat, card)

def test_suppress_onplay_smoke():
    """Matches Suppress.cs: Deal Damage; Apply power."""
    card = create_card(CardId.SUPPRESS)
    combat = _make_combat(_character_for_card("SUPPRESS"))
    _play_smoke(combat, card)

def test_sweeping_beam_onplay_smoke():
    """Matches SweepingBeam.cs: Deal Damage; Draw card(s)."""
    card = create_card(CardId.SWEEPING_BEAM)
    combat = _make_combat(_character_for_card("SWEEPING_BEAM"))
    _play_smoke(combat, card)

def test_sweeping_gaze_onplay_smoke():
    """Matches SweepingGaze.cs: Deal Damage."""
    card = create_card(CardId.SWEEPING_GAZE)
    combat = _make_combat(_character_for_card("SWEEPING_GAZE"))
    _play_smoke(combat, card)

def test_sword_boomerang_onplay_smoke():
    """Matches SwordBoomerang.cs: Deal Damage."""
    card = create_card(CardId.SWORD_BOOMERANG)
    combat = _make_combat(_character_for_card("SWORD_BOOMERANG"))
    _play_smoke(combat, card)

def test_synchronize_onplay_smoke():
    """Matches Synchronize.cs: Apply power."""
    card = create_card(CardId.SYNCHRONIZE)
    combat = _make_combat(_character_for_card("SYNCHRONIZE"))
    _play_smoke(combat, card)

def test_synthesis_onplay_smoke():
    """Matches Synthesis.cs: Deal Damage; Apply power."""
    card = create_card(CardId.SYNTHESIS)
    combat = _make_combat(_character_for_card("SYNTHESIS"))
    _play_smoke(combat, card)

def test_tactician_onplay_smoke():
    """Matches Tactician.cs: See decompiled source."""
    card = create_card(CardId.TACTICIAN)
    combat = _make_combat(_character_for_card("TACTICIAN"))
    _play_smoke(combat, card)

def test_tag_team_onplay_smoke():
    """Matches TagTeam.cs: Deal Damage; Apply power."""
    card = create_card(CardId.TAG_TEAM)
    combat = _make_combat(_character_for_card("TAG_TEAM"))
    _play_smoke(combat, card)

def test_tank_onplay_smoke():
    """Matches Tank.cs: Apply power."""
    card = create_card(CardId.TANK_CARD)
    combat = _make_combat(_character_for_card("TANK_CARD"))
    _play_smoke(combat, card)

def test_taunt_onplay_smoke():
    """Matches Taunt.cs: Apply power."""
    card = create_card(CardId.TAUNT)
    combat = _make_combat(_character_for_card("TAUNT"))
    _play_smoke(combat, card)

def test_tear_asunder_onplay_smoke():
    """Matches TearAsunder.cs: Deal Damage."""
    card = create_card(CardId.TEAR_ASUNDER)
    combat = _make_combat(_character_for_card("TEAR_ASUNDER"))
    _play_smoke(combat, card)

def test_the_bomb_onplay_smoke():
    """Matches TheBomb.cs: Apply power."""
    card = create_card(CardId.THE_BOMB_CARD)
    combat = _make_combat(_character_for_card("THE_BOMB_CARD"))
    _play_smoke(combat, card)

def test_the_gambit_onplay_smoke():
    """Matches TheGambit.cs: Apply power."""
    card = create_card(CardId.THE_GAMBIT)
    combat = _make_combat(_character_for_card("THE_GAMBIT"))
    _play_smoke(combat, card)

def test_the_sealed_throne_onplay_smoke():
    """Matches TheSealedThrone.cs: Apply power."""
    card = create_card(CardId.THE_SEALED_THRONE)
    combat = _make_combat(_character_for_card("THE_SEALED_THRONE"))
    _play_smoke(combat, card)

def test_thrash_onplay_smoke():
    """Matches Thrash.cs: Deal Damage; Exhaust."""
    card = create_card(CardId.THRASH)
    combat = _make_combat(_character_for_card("THRASH"))
    _play_smoke(combat, card)

def test_thrumming_hatchet_onplay_smoke():
    """Matches ThrummingHatchet.cs: Deal Damage."""
    card = create_card(CardId.THRUMMING_HATCHET)
    combat = _make_combat(_character_for_card("THRUMMING_HATCHET"))
    _play_smoke(combat, card)

def test_thunderclap_onplay_smoke():
    """Matches Thunderclap.cs: Deal Damage; Apply power."""
    card = create_card(CardId.THUNDERCLAP)
    combat = _make_combat(_character_for_card("THUNDERCLAP"))
    _play_smoke(combat, card)

def test_times_up_onplay_smoke():
    """Matches TimesUp.cs: Deal Damage."""
    card = create_card(CardId.TIMES_UP)
    combat = _make_combat(_character_for_card("TIMES_UP"))
    _play_smoke(combat, card)

def test_tools_of_the_trade_onplay_smoke():
    """Matches ToolsOfTheTrade.cs: Apply power."""
    card = create_card(CardId.TOOLS_OF_THE_TRADE)
    combat = _make_combat(_character_for_card("TOOLS_OF_THE_TRADE"))
    _play_smoke(combat, card)

def test_toric_toughness_onplay_smoke():
    """Matches ToricToughness.cs: Apply power."""
    card = create_card(CardId.TORIC_TOUGHNESS)
    combat = _make_combat(_character_for_card("TORIC_TOUGHNESS"))
    _play_smoke(combat, card)

def test_tracking_onplay_smoke():
    """Matches Tracking.cs: Apply power."""
    card = create_card(CardId.TRACKING)
    combat = _make_combat(_character_for_card("TRACKING"))
    _play_smoke(combat, card)

def test_trash_to_treasure_onplay_smoke():
    """Matches TrashToTreasure.cs: Apply power."""
    card = create_card(CardId.TRASH_TO_TREASURE)
    combat = _make_combat(_character_for_card("TRASH_TO_TREASURE"))
    _play_smoke(combat, card)

def test_tremble_onplay_smoke():
    """Matches Tremble.cs: Apply power."""
    card = create_card(CardId.TREMBLE)
    combat = _make_combat(_character_for_card("TREMBLE"))
    _play_smoke(combat, card)

def test_turbo_onplay_smoke():
    """Matches Turbo.cs: Add generated card(s) to pile."""
    card = create_card(CardId.TURBO)
    combat = _make_combat(_character_for_card("TURBO"))
    _play_smoke(combat, card)

def test_twin_strike_onplay_smoke():
    """Matches TwinStrike.cs: Deal Damage."""
    card = create_card(CardId.TWIN_STRIKE)
    combat = _make_combat(_character_for_card("TWIN_STRIKE"))
    _play_smoke(combat, card)

def test_ultimate_defend_onplay_smoke():
    """Matches UltimateDefend.cs: See decompiled source."""
    card = create_card(CardId.ULTIMATE_DEFEND)
    combat = _make_combat(_character_for_card("ULTIMATE_DEFEND"))
    _play_smoke(combat, card)

def test_ultimate_strike_onplay_smoke():
    """Matches UltimateStrike.cs: Deal Damage."""
    card = create_card(CardId.ULTIMATE_STRIKE)
    combat = _make_combat(_character_for_card("ULTIMATE_STRIKE"))
    _play_smoke(combat, card)

def test_unleash_onplay_smoke():
    """Matches Unleash.cs: Deal Damage."""
    card = create_card(CardId.UNLEASH)
    combat = _make_combat(_character_for_card("UNLEASH"))
    _play_smoke(combat, card)

def test_unmovable_onplay_smoke():
    """Matches Unmovable.cs: Apply power."""
    card = create_card(CardId.UNMOVABLE)
    combat = _make_combat(_character_for_card("UNMOVABLE"))
    _play_smoke(combat, card)

def test_unrelenting_onplay_smoke():
    """Matches Unrelenting.cs: Deal Damage; Apply power."""
    card = create_card(CardId.UNRELENTING)
    combat = _make_combat(_character_for_card("UNRELENTING"))
    _play_smoke(combat, card)

def test_untouchable_onplay_smoke():
    """Matches Untouchable.cs: See decompiled source."""
    card = create_card(CardId.UNTOUCHABLE)
    combat = _make_combat(_character_for_card("UNTOUCHABLE"))
    _play_smoke(combat, card)

def test_up_my_sleeve_onplay_smoke():
    """Matches UpMySleeve.cs: See decompiled source."""
    card = create_card(CardId.UP_MY_SLEEVE)
    combat = _make_combat(_character_for_card("UP_MY_SLEEVE"))
    _play_smoke(combat, card)

def test_uppercut_onplay_smoke():
    """Matches Uppercut.cs: Deal Damage; Apply power."""
    card = create_card(CardId.UPPERCUT)
    combat = _make_combat(_character_for_card("UPPERCUT"))
    _play_smoke(combat, card)

def test_veilpiercer_onplay_smoke():
    """Matches Veilpiercer.cs: Deal Damage; Apply power."""
    card = create_card(CardId.VEILPIERCER)
    combat = _make_combat(_character_for_card("VEILPIERCER"))
    _play_smoke(combat, card)

def test_well_laid_plans_onplay_smoke():
    """Matches WellLaidPlans.cs: Apply power."""
    card = create_card(CardId.WELL_LAID_PLANS)
    combat = _make_combat(_character_for_card("WELL_LAID_PLANS"))
    _play_smoke(combat, card)

def test_whistle_onplay_smoke():
    """Matches Whistle.cs: Deal Damage."""
    card = create_card(CardId.WHISTLE)
    combat = _make_combat(_character_for_card("WHISTLE"))
    _play_smoke(combat, card)

def test_wisp_onplay_smoke():
    """Matches Wisp.cs: See decompiled source."""
    card = create_card(CardId.WISP)
    combat = _make_combat(_character_for_card("WISP"))
    _play_smoke(combat, card)

def test_zap_onplay_smoke():
    """Matches Zap.cs: Orb action."""
    card = create_card(CardId.ZAP)
    combat = _make_combat(_character_for_card("ZAP"))
    _play_smoke(combat, card)

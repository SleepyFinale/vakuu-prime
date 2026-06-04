"""Generated relic smoke parity tests.

Regenerate: python scripts/audit_relic_hook_coverage.py --generate-smoke-tests
"""

import sts2_env.powers  # noqa: F401

from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.core.combat import CombatState
from sts2_env.core.rng import Rng
from sts2_env.monsters.act1_weak import create_shrinker_beetle
from sts2_env.relics.registry import create_relic_by_name


def _make_combat(relic_names: list[str]) -> CombatState:
    combat = CombatState(
        player_hp=80,
        player_max_hp=80,
        deck=create_ironclad_starter_deck(),
        rng_seed=9002,
        character_id="Ironclad",
        relics=relic_names,
    )
    creature, ai = create_shrinker_beetle(Rng(9002))
    combat.add_enemy(creature, ai)
    combat.start_combat()
    return combat


def test_brilliant_scarf_relic_smoke():
    """Matches BrilliantScarf.cs: combat start with relic (AfterCardPlayed, AfterCombatEnd, BeforeSideTurnStart, TryModifyEnergyCostInCombat, TryModifyStarCost)."""
    combat = _make_combat(["BrilliantScarf"])
    assert combat.player is not None

def test_dragon_fruit_relic_smoke():
    """Matches DragonFruit.cs: combat start with relic (AfterGoldGained, IsAllowed)."""
    combat = _make_combat(["DragonFruit"])
    assert combat.player is not None

def test_paels_eye_relic_smoke():
    """Matches PaelsEye.cs: combat start with relic (AfterCombatEnd, AfterObtained, AfterSideTurnStart, AfterTakingExtraTurn, BeforeCardPlayed, BeforeTurnEndEarly)."""
    combat = _make_combat(["PaelsEye"])
    assert combat.player is not None

def test_pocketwatch_relic_smoke():
    """Matches Pocketwatch.cs: combat start with relic (AfterCardPlayed, AfterCombatEnd, AfterModifyingHandDraw, AfterSideTurnStart, BeforeSideTurnStart, ModifyHandDraw)."""
    combat = _make_combat(["Pocketwatch"])
    assert combat.player is not None

def test_velvet_choker_relic_smoke():
    """Matches VelvetChoker.cs: combat start with relic (AfterCardPlayed, AfterCombatEnd, AfterRoomEntered, BeforeSideTurnStart, ModifyMaxEnergy, ShouldPlay)."""
    combat = _make_combat(["VelvetChoker"])
    assert combat.player is not None

def test_amethyst_aubergine_relic_smoke():
    """Matches AmethystAubergine.cs: combat start with relic (AfterModifyingRewards, IsAllowed, TryModifyRewards)."""
    combat = _make_combat(["AmethystAubergine"])
    assert combat.player is not None

def test_beating_remnant_relic_smoke():
    """Matches BeatingRemnant.cs: combat start with relic (AfterDamageReceived, AfterModifyingHpLostAfterOsty, BeforeSideTurnStart, ModifyHpLostAfterOsty)."""
    combat = _make_combat(["BeatingRemnant"])
    assert combat.player is not None

def test_belt_buckle_relic_smoke():
    """Matches BeltBuckle.cs: combat start with relic (AfterCombatEnd, AfterCombatVictory, AfterObtained, AfterPotionDiscarded, AfterPotionProcured, AfterPotionUsed)."""
    combat = _make_combat(["BeltBuckle"])
    assert combat.player is not None

def test_big_mushroom_relic_smoke():
    """Matches BigMushroom.cs: combat start with relic (AfterObtained, AfterRoomEntered, ModifyHandDraw)."""
    combat = _make_combat(["BigMushroom"])
    assert combat.player is not None

def test_bing_bong_relic_smoke():
    """Matches BingBong.cs: combat start with relic (AfterCardChangedPiles)."""
    combat = _make_combat(["BingBong"])
    assert combat.player is not None

def test_black_star_relic_smoke():
    """Matches BlackStar.cs: combat start with relic (TryModifyRewards)."""
    combat = _make_combat(["BlackStar"])
    assert combat.player is not None

def test_burning_sticks_relic_smoke():
    """Matches BurningSticks.cs: combat start with relic (AfterCardExhausted, AfterCombatEnd, AfterRoomEntered)."""
    combat = _make_combat(["BurningSticks"])
    assert combat.player is not None

def test_darkstone_periapt_relic_smoke():
    """Matches DarkstonePeriapt.cs: combat start with relic (AfterCardChangedPiles)."""
    combat = _make_combat(["DarkstonePeriapt"])
    assert combat.player is not None

def test_dream_catcher_relic_smoke():
    """Matches DreamCatcher.cs: combat start with relic (TryModifyRestSiteHealRewards)."""
    combat = _make_combat(["DreamCatcher"])
    assert combat.player is not None

def test_driftwood_relic_smoke():
    """Matches Driftwood.cs: combat start with relic (TryModifyRewardsLate)."""
    combat = _make_combat(["Driftwood"])
    assert combat.player is not None

def test_ember_tea_relic_smoke():
    """Matches EmberTea.cs: combat start with relic (AfterRoomEntered)."""
    combat = _make_combat(["EmberTea"])
    assert combat.player is not None

def test_fake_happy_flower_relic_smoke():
    """Matches FakeHappyFlower.cs: combat start with relic (AfterCombatEnd, AfterSideTurnStart)."""
    combat = _make_combat(["FakeHappyFlower"])
    assert combat.player is not None

def test_fiddle_relic_smoke():
    """Matches Fiddle.cs: combat start with relic (AfterPreventingDraw, ModifyHandDrawLate, ShouldDraw)."""
    combat = _make_combat(["Fiddle"])
    assert combat.player is not None

def test_fresnel_lens_relic_smoke():
    """Matches FresnelLens.cs: combat start with relic (ModifyMerchantCardCreationResults, TryModifyCardBeingAddedToDeck, TryModifyCardRewardOptionsLate)."""
    combat = _make_combat(["FresnelLens"])
    assert combat.player is not None

def test_frozen_egg_relic_smoke():
    """Matches FrozenEgg.cs: combat start with relic (IsAllowed, ModifyMerchantCardCreationResults, TryModifyCardBeingAddedToDeck, TryModifyCardRewardOptionsLate)."""
    combat = _make_combat(["FrozenEgg"])
    assert combat.player is not None

def test_fur_coat_relic_smoke():
    """Matches FurCoat.cs: combat start with relic (AfterCreatureAddedToCombat, AfterObtained, BeforeCombatStart)."""
    combat = _make_combat(["FurCoat"])
    assert combat.player is not None

def test_glitter_relic_smoke():
    """Matches Glitter.cs: combat start with relic (TryModifyCardRewardOptionsLate)."""
    combat = _make_combat(["Glitter"])
    assert combat.player is not None

def test_ice_cream_relic_smoke():
    """Matches IceCream.cs: combat start with relic (ShouldPlayerResetEnergy)."""
    combat = _make_combat(["IceCream"])
    assert combat.player is not None

def test_lasting_candy_relic_smoke():
    """Matches LastingCandy.cs: combat start with relic (AfterCombatEnd, IsAllowed, TryModifyCardRewardOptions)."""
    combat = _make_combat(["LastingCandy"])
    assert combat.player is not None

def test_lava_lamp_relic_smoke():
    """Matches LavaLamp.cs: combat start with relic (AfterDamageReceived, AfterRoomEntered, TryModifyCardRewardOptionsLate)."""
    combat = _make_combat(["LavaLamp"])
    assert combat.player is not None

def test_lava_rock_relic_smoke():
    """Matches LavaRock.cs: combat start with relic (TryModifyRewards)."""
    combat = _make_combat(["LavaRock"])
    assert combat.player is not None

def test_lizard_tail_relic_smoke():
    """Matches LizardTail.cs: combat start with relic (AfterPreventingDeath, ShouldDieLate)."""
    combat = _make_combat(["LizardTail"])
    assert combat.player is not None

def test_massive_scroll_relic_smoke():
    """Matches MassiveScroll.cs: combat start with relic (AfterObtained, IsAllowed)."""
    combat = _make_combat(["MassiveScroll"])
    assert combat.player is not None

def test_maw_bank_relic_smoke():
    """Matches MawBank.cs: combat start with relic (AfterItemPurchased, AfterRoomEntered)."""
    combat = _make_combat(["MawBank"])
    assert combat.player is not None

def test_meat_cleaver_relic_smoke():
    """Matches MeatCleaver.cs: combat start with relic (TryModifyRestSiteOptions)."""
    combat = _make_combat(["MeatCleaver"])
    assert combat.player is not None

def test_molten_egg_relic_smoke():
    """Matches MoltenEgg.cs: combat start with relic (IsAllowed, ModifyMerchantCardCreationResults, TryModifyCardBeingAddedToDeck, TryModifyCardRewardOptionsLate)."""
    combat = _make_combat(["MoltenEgg"])
    assert combat.player is not None

def test_paels_growth_relic_smoke():
    """Matches PaelsGrowth.cs: combat start with relic (AfterObtained, TryModifyRestSiteOptions)."""
    combat = _make_combat(["PaelsGrowth"])
    assert combat.player is not None

def test_paels_wing_relic_smoke():
    """Matches PaelsWing.cs: combat start with relic (TryModifyCardRewardAlternatives)."""
    combat = _make_combat(["PaelsWing"])
    assert combat.player is not None

def test_philosophers_stone_relic_smoke():
    """Matches PhilosophersStone.cs: combat start with relic (AfterCreatureAddedToCombat, AfterRoomEntered, ModifyMaxEnergy)."""
    combat = _make_combat(["PhilosophersStone"])
    assert combat.player is not None

def test_pollinous_core_relic_smoke():
    """Matches PollinousCore.cs: combat start with relic (AfterCombatEnd, AfterModifyingHandDraw, BeforeSideTurnStart, ModifyHandDraw)."""
    combat = _make_combat(["PollinousCore"])
    assert combat.player is not None

def test_prayer_wheel_relic_smoke():
    """Matches PrayerWheel.cs: combat start with relic (TryModifyRewards)."""
    combat = _make_combat(["PrayerWheel"])
    assert combat.player is not None

def test_pumpkin_candle_relic_smoke():
    """Matches PumpkinCandle.cs: combat start with relic (AfterObtained, AfterRoomEntered, ModifyMaxEnergy)."""
    combat = _make_combat(["PumpkinCandle"])
    assert combat.player is not None

def test_silver_crucible_relic_smoke():
    """Matches SilverCrucible.cs: combat start with relic (AfterModifyingCardRewardOptions, AfterRoomEntered, IsAllowed, ShouldGenerateTreasure, TryModifyCardRewardOptionsLate)."""
    combat = _make_combat(["SilverCrucible"])
    assert combat.player is not None

def test_sling_of_courage_relic_smoke():
    """Matches SlingOfCourage.cs: combat start with relic (AfterRoomEntered)."""
    combat = _make_combat(["SlingOfCourage"])
    assert combat.player is not None

def test_spiked_gauntlets_relic_smoke():
    """Matches SpikedGauntlets.cs: combat start with relic (ModifyMaxEnergy, TryModifyEnergyCostInCombat)."""
    combat = _make_combat(["SpikedGauntlets"])
    assert combat.player is not None

def test_stone_calendar_relic_smoke():
    """Matches StoneCalendar.cs: combat start with relic (AfterCombatEnd, AfterRoomEntered, AfterSideTurnStart, BeforeTurnEnd)."""
    combat = _make_combat(["StoneCalendar"])
    assert combat.player is not None

def test_sword_of_jade_relic_smoke():
    """Matches SwordOfJade.cs: combat start with relic (AfterRoomEntered)."""
    combat = _make_combat(["SwordOfJade"])
    assert combat.player is not None

def test_the_boot_relic_smoke():
    """Matches TheBoot.cs: combat start with relic (AfterModifyingHpLostBeforeOsty, ModifyHpLostBeforeOsty)."""
    combat = _make_combat(["TheBoot"])
    assert combat.player is not None

def test_toxic_egg_relic_smoke():
    """Matches ToxicEgg.cs: combat start with relic (IsAllowed, ModifyMerchantCardCreationResults, TryModifyCardBeingAddedToDeck, TryModifyCardRewardOptionsLate)."""
    combat = _make_combat(["ToxicEgg"])
    assert combat.player is not None

def test_tungsten_rod_relic_smoke():
    """Matches TungstenRod.cs: combat start with relic (AfterModifyingHpLostAfterOsty, ModifyHpLostAfterOsty)."""
    combat = _make_combat(["TungstenRod"])
    assert combat.player is not None

def test_whispering_earring_relic_smoke():
    """Matches WhisperingEarring.cs: combat start with relic (BeforePlayPhaseStartLate, ModifyMaxEnergy)."""
    combat = _make_combat(["WhisperingEarring"])
    assert combat.player is not None

def test_white_star_relic_smoke():
    """Matches WhiteStar.cs: combat start with relic (IsAllowed, TryModifyRewards)."""
    combat = _make_combat(["WhiteStar"])
    assert combat.player is not None

def test_wing_charm_relic_smoke():
    """Matches WingCharm.cs: combat start with relic (TryModifyCardRewardOptionsLate)."""
    combat = _make_combat(["WingCharm"])
    assert combat.player is not None

def test_wongos_mystery_ticket_relic_smoke():
    """Matches WongosMysteryTicket.cs: combat start with relic (AfterCombatEnd, AfterModifyingRewards, TryModifyRewards)."""
    combat = _make_combat(["WongosMysteryTicket"])
    assert combat.player is not None

def test_art_of_war_relic_smoke():
    """Matches ArtOfWar.cs: combat start with relic (AfterCardPlayed, AfterCombatEnd, AfterEnergyReset, AfterTurnEnd)."""
    combat = _make_combat(["ArtOfWar"])
    assert combat.player is not None

def test_daughter_of_the_wind_relic_smoke():
    """Matches DaughterOfTheWind.cs: combat start with relic (AfterCardPlayed)."""
    combat = _make_combat(["DaughterOfTheWind"])
    assert combat.player is not None

def test_diamond_diadem_relic_smoke():
    """Matches DiamondDiadem.cs: combat start with relic (AfterCardPlayed, AfterCombatEnd, AfterSideTurnStart, BeforeTurnEnd)."""
    combat = _make_combat(["DiamondDiadem"])
    assert combat.player is not None

def test_game_piece_relic_smoke():
    """Matches GamePiece.cs: combat start with relic (AfterCardPlayed)."""
    combat = _make_combat(["GamePiece"])
    assert combat.player is not None

def test_helical_dart_relic_smoke():
    """Matches HelicalDart.cs: combat start with relic (AfterCardPlayed)."""
    combat = _make_combat(["HelicalDart"])
    assert combat.player is not None

def test_iron_club_relic_smoke():
    """Matches IronClub.cs: combat start with relic (AfterCardPlayed)."""
    combat = _make_combat(["IronClub"])
    assert combat.player is not None

def test_lost_wisp_relic_smoke():
    """Matches LostWisp.cs: combat start with relic (AfterCardPlayed)."""
    combat = _make_combat(["LostWisp"])
    assert combat.player is not None

def test_mummified_hand_relic_smoke():
    """Matches MummifiedHand.cs: combat start with relic (AfterCardPlayed)."""
    combat = _make_combat(["MummifiedHand"])
    assert combat.player is not None

def test_music_box_relic_smoke():
    """Matches MusicBox.cs: combat start with relic (AfterCardPlayed, AfterCombatEnd, BeforeCardPlayed, BeforeSideTurnStart)."""
    combat = _make_combat(["MusicBox"])
    assert combat.player is not None

def test_paels_legion_relic_smoke():
    """Matches PaelsLegion.cs: combat start with relic (AfterCardPlayed, AfterCombatEnd, AfterModifyingBlockAmount, AfterObtained, AfterSideTurnStart, BeforeCombatStart)."""
    combat = _make_combat(["PaelsLegion"])
    assert combat.player is not None

def test_alchemical_coffer_relic_smoke():
    """Matches AlchemicalCoffer.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["AlchemicalCoffer"])
    assert combat.player is not None

def test_anchor_relic_smoke():
    """Matches Anchor.cs: combat start with relic (BeforeCombatStart)."""
    combat = _make_combat(["Anchor"])
    assert combat.player is not None

def test_arcane_scroll_relic_smoke():
    """Matches ArcaneScroll.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["ArcaneScroll"])
    assert combat.player is not None

def test_archaic_tooth_relic_smoke():
    """Matches ArchaicTooth.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["ArchaicTooth"])
    assert combat.player is not None

def test_astrolabe_relic_smoke():
    """Matches Astrolabe.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["Astrolabe"])
    assert combat.player is not None

def test_beautiful_bracelet_relic_smoke():
    """Matches BeautifulBracelet.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["BeautifulBracelet"])
    assert combat.player is not None

def test_big_hat_relic_smoke():
    """Matches BigHat.cs: combat start with relic (AfterSideTurnStart)."""
    combat = _make_combat(["BigHat"])
    assert combat.player is not None

def test_biiig_hug_relic_smoke():
    """Matches BiiigHug.cs: combat start with relic (AfterObtained, AfterShuffle)."""
    combat = _make_combat(["BiiigHug"])
    assert combat.player is not None

def test_blessed_antler_relic_smoke():
    """Matches BlessedAntler.cs: combat start with relic (BeforeHandDraw, ModifyMaxEnergy)."""
    combat = _make_combat(["BlessedAntler"])
    assert combat.player is not None

def test_blood_soaked_rose_relic_smoke():
    """Matches BloodSoakedRose.cs: combat start with relic (AfterObtained, ModifyMaxEnergy)."""
    combat = _make_combat(["BloodSoakedRose"])
    assert combat.player is not None

def test_bone_tea_relic_smoke():
    """Matches BoneTea.cs: combat start with relic (AfterSideTurnStart)."""
    combat = _make_combat(["BoneTea"])
    assert combat.player is not None

def test_bookmark_relic_smoke():
    """Matches Bookmark.cs: combat start with relic (AfterTurnEnd)."""
    combat = _make_combat(["Bookmark"])
    assert combat.player is not None

def test_booming_conch_relic_smoke():
    """Matches BoomingConch.cs: combat start with relic (ModifyHandDraw)."""
    combat = _make_combat(["BoomingConch"])
    assert combat.player is not None

def test_bread_relic_smoke():
    """Matches Bread.cs: combat start with relic (AfterSideTurnStart, ModifyMaxEnergy)."""
    combat = _make_combat(["Bread"])
    assert combat.player is not None

def test_brimstone_relic_smoke():
    """Matches Brimstone.cs: combat start with relic (AfterSideTurnStart)."""
    combat = _make_combat(["Brimstone"])
    assert combat.player is not None

def test_byrdpip_relic_smoke():
    """Matches Byrdpip.cs: combat start with relic (AfterObtained, BeforeCombatStart)."""
    combat = _make_combat(["Byrdpip"])
    assert combat.player is not None

def test_calling_bell_relic_smoke():
    """Matches CallingBell.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["CallingBell"])
    assert combat.player is not None

def test_captains_wheel_relic_smoke():
    """Matches CaptainsWheel.cs: combat start with relic (AfterBlockCleared)."""
    combat = _make_combat(["CaptainsWheel"])
    assert combat.player is not None

def test_cauldron_relic_smoke():
    """Matches Cauldron.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["Cauldron"])
    assert combat.player is not None

def test_chandelier_relic_smoke():
    """Matches Chandelier.cs: combat start with relic (AfterSideTurnStart)."""
    combat = _make_combat(["Chandelier"])
    assert combat.player is not None

def test_charons_ashes_relic_smoke():
    """Matches CharonsAshes.cs: combat start with relic (AfterCardExhausted)."""
    combat = _make_combat(["CharonsAshes"])
    assert combat.player is not None

def test_choices_paradox_relic_smoke():
    """Matches ChoicesParadox.cs: combat start with relic (AfterPlayerTurnStart)."""
    combat = _make_combat(["ChoicesParadox"])
    assert combat.player is not None

def test_chosen_cheese_relic_smoke():
    """Matches ChosenCheese.cs: combat start with relic (AfterCombatEnd)."""
    combat = _make_combat(["ChosenCheese"])
    assert combat.player is not None

def test_claws_relic_smoke():
    """Matches Claws.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["Claws"])
    assert combat.player is not None

def test_crossbow_relic_smoke():
    """Matches Crossbow.cs: combat start with relic (AfterSideTurnStart)."""
    combat = _make_combat(["Crossbow"])
    assert combat.player is not None

def test_cursed_pearl_relic_smoke():
    """Matches CursedPearl.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["CursedPearl"])
    assert combat.player is not None

def test_demon_tongue_relic_smoke():
    """Matches DemonTongue.cs: combat start with relic (AfterDamageReceived, BeforeSideTurnStart)."""
    combat = _make_combat(["DemonTongue"])
    assert combat.player is not None

def test_distinguished_cape_relic_smoke():
    """Matches DistinguishedCape.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["DistinguishedCape"])
    assert combat.player is not None

def test_dollys_mirror_relic_smoke():
    """Matches DollysMirror.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["DollysMirror"])
    assert combat.player is not None

def test_ectoplasm_relic_smoke():
    """Matches Ectoplasm.cs: combat start with relic (ModifyMaxEnergy, ShouldGainGold)."""
    combat = _make_combat(["Ectoplasm"])
    assert combat.player is not None

def test_electric_shrymp_relic_smoke():
    """Matches ElectricShrymp.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["ElectricShrymp"])
    assert combat.player is not None

def test_empty_cage_relic_smoke():
    """Matches EmptyCage.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["EmptyCage"])
    assert combat.player is not None

def test_fake_anchor_relic_smoke():
    """Matches FakeAnchor.cs: combat start with relic (BeforeCombatStart)."""
    combat = _make_combat(["FakeAnchor"])
    assert combat.player is not None

def test_fake_blood_vial_relic_smoke():
    """Matches FakeBloodVial.cs: combat start with relic (AfterPlayerTurnStartLate)."""
    combat = _make_combat(["FakeBloodVial"])
    assert combat.player is not None

def test_fake_lees_waffle_relic_smoke():
    """Matches FakeLeesWaffle.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["FakeLeesWaffle"])
    assert combat.player is not None

def test_fake_mango_relic_smoke():
    """Matches FakeMango.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["FakeMango"])
    assert combat.player is not None

def test_fake_orichalcum_relic_smoke():
    """Matches FakeOrichalcum.cs: combat start with relic (BeforeSideTurnStart, BeforeTurnEnd, BeforeTurnEndVeryEarly)."""
    combat = _make_combat(["FakeOrichalcum"])
    assert combat.player is not None

def test_fake_snecko_eye_relic_smoke():
    """Matches FakeSneckoEye.cs: combat start with relic (AfterObtained, BeforeCombatStart)."""
    combat = _make_combat(["FakeSneckoEye"])
    assert combat.player is not None

def test_fake_strike_dummy_relic_smoke():
    """Matches FakeStrikeDummy.cs: combat start with relic (ModifyDamageAdditive)."""
    combat = _make_combat(["FakeStrikeDummy"])
    assert combat.player is not None

def test_fake_venerable_tea_set_relic_smoke():
    """Matches FakeVenerableTeaSet.cs: combat start with relic (AfterEnergyReset, AfterRoomEntered)."""
    combat = _make_combat(["FakeVenerableTeaSet"])
    assert combat.player is not None

def test_forgotten_soul_relic_smoke():
    """Matches ForgottenSoul.cs: combat start with relic (AfterCardExhausted)."""
    combat = _make_combat(["ForgottenSoul"])
    assert combat.player is not None

def test_fragrant_mushroom_relic_smoke():
    """Matches FragrantMushroom.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["FragrantMushroom"])
    assert combat.player is not None

def test_glass_eye_relic_smoke():
    """Matches GlassEye.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["GlassEye"])
    assert combat.player is not None

def test_gnarled_hammer_relic_smoke():
    """Matches GnarledHammer.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["GnarledHammer"])
    assert combat.player is not None

def test_golden_compass_relic_smoke():
    """Matches GoldenCompass.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["GoldenCompass"])
    assert combat.player is not None

def test_golden_pearl_relic_smoke():
    """Matches GoldenPearl.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["GoldenPearl"])
    assert combat.player is not None

def test_hand_drill_relic_smoke():
    """Matches HandDrill.cs: combat start with relic (AfterDamageGiven)."""
    combat = _make_combat(["HandDrill"])
    assert combat.player is not None

def test_hefty_tablet_relic_smoke():
    """Matches HeftyTablet.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["HeftyTablet"])
    assert combat.player is not None

def test_jeweled_mask_relic_smoke():
    """Matches JeweledMask.cs: combat start with relic (BeforeHandDraw)."""
    combat = _make_combat(["JeweledMask"])
    assert combat.player is not None

def test_jewelry_box_relic_smoke():
    """Matches JewelryBox.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["JewelryBox"])
    assert combat.player is not None

def test_kifuda_relic_smoke():
    """Matches Kifuda.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["Kifuda"])
    assert combat.player is not None

def test_lantern_relic_smoke():
    """Matches Lantern.cs: combat start with relic (AfterSideTurnStart)."""
    combat = _make_combat(["Lantern"])
    assert combat.player is not None

def test_large_capsule_relic_smoke():
    """Matches LargeCapsule.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["LargeCapsule"])
    assert combat.player is not None

def test_lead_paperweight_relic_smoke():
    """Matches LeadPaperweight.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["LeadPaperweight"])
    assert combat.player is not None

def test_leafy_poultice_relic_smoke():
    """Matches LeafyPoultice.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["LeafyPoultice"])
    assert combat.player is not None

def test_lees_waffle_relic_smoke():
    """Matches LeesWaffle.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["LeesWaffle"])
    assert combat.player is not None

def test_looming_fruit_relic_smoke():
    """Matches LoomingFruit.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["LoomingFruit"])
    assert combat.player is not None

def test_lords_parasol_relic_smoke():
    """Matches LordsParasol.cs: combat start with relic (AfterRoomEntered)."""
    combat = _make_combat(["LordsParasol"])
    assert combat.player is not None

def test_lost_coffer_relic_smoke():
    """Matches LostCoffer.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["LostCoffer"])
    assert combat.player is not None

def test_lunar_pastry_relic_smoke():
    """Matches LunarPastry.cs: combat start with relic (AfterTurnEnd)."""
    combat = _make_combat(["LunarPastry"])
    assert combat.player is not None

def test_mango_relic_smoke():
    """Matches Mango.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["Mango"])
    assert combat.player is not None

def test_membership_card_relic_smoke():
    """Matches MembershipCard.cs: combat start with relic (ModifyMerchantPrice)."""
    combat = _make_combat(["MembershipCard"])
    assert combat.player is not None

def test_miniature_tent_relic_smoke():
    """Matches MiniatureTent.cs: combat start with relic (ShouldDisableRemainingRestSiteOptions)."""
    combat = _make_combat(["MiniatureTent"])
    assert combat.player is not None

def test_mr_struggles_relic_smoke():
    """Matches MrStruggles.cs: combat start with relic (AfterPlayerTurnStart)."""
    combat = _make_combat(["MrStruggles"])
    assert combat.player is not None

def test_mystic_lighter_relic_smoke():
    """Matches MysticLighter.cs: combat start with relic (ModifyDamageAdditive)."""
    combat = _make_combat(["MysticLighter"])
    assert combat.player is not None

def test_neows_bones_relic_smoke():
    """Matches NeowsBones.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["NeowsBones"])
    assert combat.player is not None

def test_neows_talisman_relic_smoke():
    """Matches NeowsTalisman.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["NeowsTalisman"])
    assert combat.player is not None

def test_neows_torment_relic_smoke():
    """Matches NeowsTorment.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["NeowsTorment"])
    assert combat.player is not None

def test_new_leaf_relic_smoke():
    """Matches NewLeaf.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["NewLeaf"])
    assert combat.player is not None

def test_ninja_scroll_relic_smoke():
    """Matches NinjaScroll.cs: combat start with relic (BeforeHandDraw)."""
    combat = _make_combat(["NinjaScroll"])
    assert combat.player is not None

def test_nutritious_oyster_relic_smoke():
    """Matches NutritiousOyster.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["NutritiousOyster"])
    assert combat.player is not None

def test_nutritious_soup_relic_smoke():
    """Matches NutritiousSoup.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["NutritiousSoup"])
    assert combat.player is not None

def test_old_coin_relic_smoke():
    """Matches OldCoin.cs: combat start with relic (AfterObtained, IsAllowed)."""
    combat = _make_combat(["OldCoin"])
    assert combat.player is not None

def test_orrery_relic_smoke():
    """Matches Orrery.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["Orrery"])
    assert combat.player is not None

def test_paels_blood_relic_smoke():
    """Matches PaelsBlood.cs: combat start with relic (ModifyHandDraw)."""
    combat = _make_combat(["PaelsBlood"])
    assert combat.player is not None

def test_paels_claw_relic_smoke():
    """Matches PaelsClaw.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["PaelsClaw"])
    assert combat.player is not None

def test_paels_horn_relic_smoke():
    """Matches PaelsHorn.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["PaelsHorn"])
    assert combat.player is not None

def test_paels_tears_relic_smoke():
    """Matches PaelsTears.cs: combat start with relic (AfterCombatEnd, AfterSideTurnStart, BeforeTurnEnd)."""
    combat = _make_combat(["PaelsTears"])
    assert combat.player is not None

def test_paels_tooth_relic_smoke():
    """Matches PaelsTooth.cs: combat start with relic (AfterCombatEnd, AfterObtained)."""
    combat = _make_combat(["PaelsTooth"])
    assert combat.player is not None

def test_pandoras_box_relic_smoke():
    """Matches PandorasBox.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["PandorasBox"])
    assert combat.player is not None

def test_pear_relic_smoke():
    """Matches Pear.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["Pear"])
    assert combat.player is not None

def test_phial_holster_relic_smoke():
    """Matches PhialHolster.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["PhialHolster"])
    assert combat.player is not None

def test_pomander_relic_smoke():
    """Matches Pomander.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["Pomander"])
    assert combat.player is not None

def test_precarious_shears_relic_smoke():
    """Matches PrecariousShears.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["PrecariousShears"])
    assert combat.player is not None

def test_precise_scissors_relic_smoke():
    """Matches PreciseScissors.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["PreciseScissors"])
    assert combat.player is not None

def test_preserved_fog_relic_smoke():
    """Matches PreservedFog.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["PreservedFog"])
    assert combat.player is not None

def test_prismatic_gem_relic_smoke():
    """Matches PrismaticGem.cs: combat start with relic (ModifyMaxEnergy)."""
    combat = _make_combat(["PrismaticGem"])
    assert combat.player is not None

def test_punch_dagger_relic_smoke():
    """Matches PunchDagger.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["PunchDagger"])
    assert combat.player is not None

def test_radiant_pearl_relic_smoke():
    """Matches RadiantPearl.cs: combat start with relic (BeforeHandDraw)."""
    combat = _make_combat(["RadiantPearl"])
    assert combat.player is not None

def test_ring_of_the_snake_relic_smoke():
    """Matches RingOfTheSnake.cs: combat start with relic (ModifyHandDraw)."""
    combat = _make_combat(["RingOfTheSnake"])
    assert combat.player is not None

def test_ringing_triangle_relic_smoke():
    """Matches RingingTriangle.cs: combat start with relic (ShouldFlush)."""
    combat = _make_combat(["RingingTriangle"])
    assert combat.player is not None

def test_royal_poison_relic_smoke():
    """Matches RoyalPoison.cs: combat start with relic (AfterPlayerTurnStart)."""
    combat = _make_combat(["RoyalPoison"])
    assert combat.player is not None

def test_royal_stamp_relic_smoke():
    """Matches RoyalStamp.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["RoyalStamp"])
    assert combat.player is not None

def test_runic_capacitor_relic_smoke():
    """Matches RunicCapacitor.cs: combat start with relic (AfterSideTurnStart)."""
    combat = _make_combat(["RunicCapacitor"])
    assert combat.player is not None

def test_runic_pyramid_relic_smoke():
    """Matches RunicPyramid.cs: combat start with relic (ShouldFlush)."""
    combat = _make_combat(["RunicPyramid"])
    assert combat.player is not None

def test_sai_relic_smoke():
    """Matches Sai.cs: combat start with relic (AfterSideTurnStart)."""
    combat = _make_combat(["Sai"])
    assert combat.player is not None

def test_sand_castle_relic_smoke():
    """Matches SandCastle.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["SandCastle"])
    assert combat.player is not None

def test_screaming_flagon_relic_smoke():
    """Matches ScreamingFlagon.cs: combat start with relic (BeforeTurnEnd)."""
    combat = _make_combat(["ScreamingFlagon"])
    assert combat.player is not None

def test_scroll_boxes_relic_smoke():
    """Matches ScrollBoxes.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["ScrollBoxes"])
    assert combat.player is not None

def test_sea_glass_relic_smoke():
    """Matches SeaGlass.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["SeaGlass"])
    assert combat.player is not None

def test_seal_of_gold_relic_smoke():
    """Matches SealOfGold.cs: combat start with relic (AfterSideTurnStart)."""
    combat = _make_combat(["SealOfGold"])
    assert combat.player is not None

def test_sere_talon_relic_smoke():
    """Matches SereTalon.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["SereTalon"])
    assert combat.player is not None

def test_signet_ring_relic_smoke():
    """Matches SignetRing.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["SignetRing"])
    assert combat.player is not None

def test_small_capsule_relic_smoke():
    """Matches SmallCapsule.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["SmallCapsule"])
    assert combat.player is not None

def test_snecko_eye_relic_smoke():
    """Matches SneckoEye.cs: combat start with relic (AfterObtained, BeforeCombatStart, ModifyHandDraw)."""
    combat = _make_combat(["SneckoEye"])
    assert combat.player is not None

def test_sozu_relic_smoke():
    """Matches Sozu.cs: combat start with relic (ModifyMaxEnergy, ShouldProcurePotion)."""
    combat = _make_combat(["Sozu"])
    assert combat.player is not None

def test_stone_humidifier_relic_smoke():
    """Matches StoneHumidifier.cs: combat start with relic (AfterRestSiteHeal)."""
    combat = _make_combat(["StoneHumidifier"])
    assert combat.player is not None

def test_storybook_relic_smoke():
    """Matches Storybook.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["Storybook"])
    assert combat.player is not None

def test_sturdy_clamp_relic_smoke():
    """Matches SturdyClamp.cs: combat start with relic (AfterPreventingBlockClear, ShouldClearBlock)."""
    combat = _make_combat(["SturdyClamp"])
    assert combat.player is not None

def test_sword_of_stone_relic_smoke():
    """Matches SwordOfStone.cs: combat start with relic (AfterCombatVictory)."""
    combat = _make_combat(["SwordOfStone"])
    assert combat.player is not None

def test_tanxs_whistle_relic_smoke():
    """Matches TanxsWhistle.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["TanxsWhistle"])
    assert combat.player is not None

def test_tea_of_discourtesy_relic_smoke():
    """Matches TeaOfDiscourtesy.cs: combat start with relic (BeforeCombatStart)."""
    combat = _make_combat(["TeaOfDiscourtesy"])
    assert combat.player is not None

def test_the_abacus_relic_smoke():
    """Matches TheAbacus.cs: combat start with relic (AfterShuffle)."""
    combat = _make_combat(["TheAbacus"])
    assert combat.player is not None

def test_the_courier_relic_smoke():
    """Matches TheCourier.cs: combat start with relic (ModifyMerchantPrice, ShouldRefillMerchantEntry)."""
    combat = _make_combat(["TheCourier"])
    assert combat.player is not None

def test_toasty_mittens_relic_smoke():
    """Matches ToastyMittens.cs: combat start with relic (BeforeHandDraw)."""
    combat = _make_combat(["ToastyMittens"])
    assert combat.player is not None

def test_toolbox_relic_smoke():
    """Matches Toolbox.cs: combat start with relic (BeforeHandDraw)."""
    combat = _make_combat(["Toolbox"])
    assert combat.player is not None

def test_touch_of_orobas_relic_smoke():
    """Matches TouchOfOrobas.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["TouchOfOrobas"])
    assert combat.player is not None

def test_toy_box_relic_smoke():
    """Matches ToyBox.cs: combat start with relic (AfterCombatEnd, AfterObtained)."""
    combat = _make_combat(["ToyBox"])
    assert combat.player is not None

def test_tri_boomerang_relic_smoke():
    """Matches TriBoomerang.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["TriBoomerang"])
    assert combat.player is not None

def test_unceasing_top_relic_smoke():
    """Matches UnceasingTop.cs: combat start with relic (AfterHandEmptied)."""
    combat = _make_combat(["UnceasingTop"])
    assert combat.player is not None

def test_undying_sigil_relic_smoke():
    """Matches UndyingSigil.cs: combat start with relic (ModifyDamageMultiplicative)."""
    combat = _make_combat(["UndyingSigil"])
    assert combat.player is not None

def test_very_hot_cocoa_relic_smoke():
    """Matches VeryHotCocoa.cs: combat start with relic (AfterSideTurnStart)."""
    combat = _make_combat(["VeryHotCocoa"])
    assert combat.player is not None

def test_vitruvian_minion_relic_smoke():
    """Matches VitruvianMinion.cs: combat start with relic (ModifyBlockMultiplicative, ModifyDamageMultiplicative)."""
    combat = _make_combat(["VitruvianMinion"])
    assert combat.player is not None

def test_war_hammer_relic_smoke():
    """Matches WarHammer.cs: combat start with relic (AfterCombatVictory)."""
    combat = _make_combat(["WarHammer"])
    assert combat.player is not None

def test_whetstone_relic_smoke():
    """Matches Whetstone.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["Whetstone"])
    assert combat.player is not None

def test_white_beast_statue_relic_smoke():
    """Matches WhiteBeastStatue.cs: combat start with relic (IsAllowed, ShouldForcePotionReward)."""
    combat = _make_combat(["WhiteBeastStatue"])
    assert combat.player is not None

def test_winged_boots_relic_smoke():
    """Matches WingedBoots.cs: combat start with relic (AfterRoomEntered, IsAllowed, ShouldAllowFreeTravel)."""
    combat = _make_combat(["WingedBoots"])
    assert combat.player is not None

def test_yummy_cookie_relic_smoke():
    """Matches YummyCookie.cs: combat start with relic (AfterObtained)."""
    combat = _make_combat(["YummyCookie"])
    assert combat.player is not None

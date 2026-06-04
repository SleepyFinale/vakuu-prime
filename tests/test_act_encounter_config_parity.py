"""Parity between acts.py encounter ID lists and encounter setup modules."""

from __future__ import annotations

import pytest

from sts2_env.encounters.pools import encounter_lists_for_act
from sts2_env.map.acts import (
    ACT_1,
    ACT_1_OVERGROWTH,
    ACT_1_UNDERDOCKS,
    ACT_2,
    ActConfig,
    BiomeId,
)

# From decompiled GenerateAllEncounters (Overgrowth, Underdocks, Hive, Glory).
DECOMPILED_ENCOUNTER_MANIFEST: dict[BiomeId, dict[str, list[str]]] = {
    "overgrowth": {
        "weak": [
            "NibbitsWeak",
            "SlimesWeak",
            "ShrinkerBeetleWeak",
            "FuzzyWurmCrawlerWeak",
        ],
        "normal": [
            "CubexConstructNormal",
            "FlyconidNormal",
            "FogmogNormal",
            "InkletsNormal",
            "MawlerNormal",
            "NibbitsNormal",
            "OvergrowthCrawlers",
            "RubyRaidersNormal",
            "SlimesNormal",
            "SlitheringStranglerNormal",
            "SnappingJaxfruitNormal",
            "VineShamblerNormal",
        ],
        "elite": [
            "BygoneEffigyElite",
            "ByrdonisElite",
            "PhrogParasiteElite",
        ],
        "boss": ["VantomBoss", "CeremonialBeastBoss", "TheKinBoss"],
    },
    "underdocks": {
        "weak": [
            "CorpseSlugsWeak",
            "SeapunkWeak",
            "SludgeSpinnerWeak",
            "ToadpolesWeak",
        ],
        "normal": [
            "CorpseSlugsNormal",
            "CultistsNormal",
            "FossilStalkerNormal",
            "GremlinMercNormal",
            "HauntedShipNormal",
            "LivingFogNormal",
            "PunchConstructNormal",
            "SeapunkNormal",
            "SewerClamNormal",
            "TwoTailedRatsNormal",
        ],
        "elite": [
            "PhantasmalGardenersElite",
            "SkulkingColonyElite",
            "TerrorEelElite",
        ],
        "boss": [
            "WaterfallGiantBoss",
            "SoulFyshBoss",
            "LagavulinMatriarchBoss",
        ],
    },
    "hive": {
        "weak": [
            "BowlbugsWeak",
            "ExoskeletonsWeak",
            "ThievingHopperWeak",
            "TunnelerWeak",
        ],
        "normal": [
            "BowlbugsNormal",
            "ChompersNormal",
            "ExoskeletonsNormal",
            "HunterKillerNormal",
            "LouseProgenitorNormal",
            "MytesNormal",
            "OvicopterNormal",
            "SlumberingBeetleNormal",
            "SpinyToadNormal",
            "TheObscuraNormal",
        ],
        "elite": [
            "DecimillipedeElite",
            "EntomancerElite",
            "InfestedPrismsElite",
        ],
        "boss": [
            "TheInsatiableBoss",
            "KnowledgeDemonBoss",
            "KaiserCrabBoss",
        ],
    },
    "glory": {
        "weak": [
            "DevotedSculptorWeak",
            "ScrollsOfBitingWeak",
            "TurretOperatorWeak",
        ],
        "normal": [
            "AxebotsNormal",
            "ConstructMenagerieNormal",
            "FabricatorNormal",
            "FrogKnightNormal",
            "GlobeHeadNormal",
            "OwlMagistrateNormal",
            "ScrollsOfBitingNormal",
            "SlimedBerserkerNormal",
            "TheLostAndForgottenNormal",
        ],
        "elite": [
            "KnightsElite",
            "MechaKnightElite",
            "SoulNexusElite",
        ],
        "boss": ["QueenBoss", "TestSubjectBoss", "DoormakerBoss"],
    },
}

_BIOME_ACT_CONFIG: dict[BiomeId, ActConfig] = {
    "overgrowth": ACT_1_OVERGROWTH,
    "underdocks": ACT_1_UNDERDOCKS,
    "hive": ACT_1,
    "glory": ACT_2,
}

_TIER_SUFFIXES = ("weak", "normal", "elite", "boss")


def setup_name_to_encounter_id(setup_name: str) -> str:
    """Map ``setup_nibbits_weak`` -> ``NibbitsWeak`` (matches C# encounter type names)."""
    assert setup_name.startswith("setup_")
    body = setup_name.removeprefix("setup_")
    parts = body.split("_")
    if parts[-1] in _TIER_SUFFIXES:
        tier = parts[-1]
        words = parts[:-1]
        tier_pascal = tier[0].upper() + tier[1:]
        word_parts = [w[0].upper() + w[1:] for w in words]
        return "".join(word_parts) + tier_pascal
    word_parts = [w[0].upper() + w[1:] for w in parts]
    return "".join(word_parts)


@pytest.mark.parametrize("biome_id", list(DECOMPILED_ENCOUNTER_MANIFEST))
class TestActsPyMatchesDecompiledManifest:
    def test_weak_encounter_ids(self, biome_id: BiomeId) -> None:
        act = _BIOME_ACT_CONFIG[biome_id]
        expected = DECOMPILED_ENCOUNTER_MANIFEST[biome_id]["weak"]
        assert act.weak_encounter_ids == expected

    def test_strong_encounter_ids(self, biome_id: BiomeId) -> None:
        act = _BIOME_ACT_CONFIG[biome_id]
        expected = DECOMPILED_ENCOUNTER_MANIFEST[biome_id]["normal"]
        assert act.strong_encounter_ids == expected

    def test_elite_ids(self, biome_id: BiomeId) -> None:
        act = _BIOME_ACT_CONFIG[biome_id]
        expected = DECOMPILED_ENCOUNTER_MANIFEST[biome_id]["elite"]
        assert act.elite_ids == expected

    def test_boss_ids(self, biome_id: BiomeId) -> None:
        act = _BIOME_ACT_CONFIG[biome_id]
        expected = DECOMPILED_ENCOUNTER_MANIFEST[biome_id]["boss"]
        assert act.boss_ids == expected


@pytest.mark.parametrize("biome_id", list(DECOMPILED_ENCOUNTER_MANIFEST))
class TestEncounterModulePoolSizes:
    def test_pool_lengths_match_manifest(self, biome_id: BiomeId) -> None:
        manifest = DECOMPILED_ENCOUNTER_MANIFEST[biome_id]
        act_index = 0 if biome_id in ("overgrowth", "underdocks") else (
            1 if biome_id == "hive" else 2
        )
        weak, normal, elite, boss = encounter_lists_for_act(
            act_index,
            biome_id=biome_id if act_index == 0 else None,
        )
        assert len(weak) == len(manifest["weak"])
        assert len(normal) == len(manifest["normal"])
        assert len(elite) == len(manifest["elite"])
        assert len(boss) == len(manifest["boss"])


@pytest.mark.parametrize("biome_id", list(DECOMPILED_ENCOUNTER_MANIFEST))
class TestSetupFunctionsMatchManifestIds:
    def test_setup_names_convert_to_manifest_ids(self, biome_id: BiomeId) -> None:
        manifest = DECOMPILED_ENCOUNTER_MANIFEST[biome_id]
        act_index = 0 if biome_id in ("overgrowth", "underdocks") else (
            1 if biome_id == "hive" else 2
        )
        weak, normal, elite, boss = encounter_lists_for_act(
            act_index,
            biome_id=biome_id if act_index == 0 else None,
        )
        for tier, pool in (
            ("weak", weak),
            ("normal", normal),
            ("elite", elite),
            ("boss", boss),
        ):
            ids_from_setups = {setup_name_to_encounter_id(fn.__name__) for fn in pool}
            assert ids_from_setups == set(manifest[tier]), (
                f"{biome_id} {tier}: setups={ids_from_setups}, manifest={set(manifest[tier])}"
            )


def test_total_encounters_across_biomes_is_eighty() -> None:
    total = sum(
        len(tier_ids)
        for biome in DECOMPILED_ENCOUNTER_MANIFEST.values()
        for tier_ids in biome.values()
    )
    assert total == 80

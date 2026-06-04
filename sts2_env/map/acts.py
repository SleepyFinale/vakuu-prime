"""Per-act configuration: room counts, encounter pools, event pools, boss pools.

Runs have 3 acts. Act 1 is either Overgrowth or Underdocks (alternate biomes, same
map rules); acts 2 and 3 are Hive and Glory. Based on decompiled ActModel.GetRandomList.

Combat encounter selection uses ``encounter_lists_for_act`` in
``sts2_env.encounters.pools`` (setup functions in act1–act4 modules). The
``weak_encounter_ids``, ``strong_encounter_ids``, and ``elite_ids`` fields here
mirror decompiled ``GenerateAllEncounters`` for documentation and parity tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from sts2_env.core.rng import Rng

# Events shared across all acts (ModelDb.AllSharedEvents). These IDs are shuffled
# into every act's event_ids at run init; biome/act eligibility is enforced per
# event via is_allowed at pick time (e.g. BrainLeech is not pickable in Glory).
SHARED_EVENT_IDS: list[str] = [
    "BrainLeech",
    "CrystalSphere",
    "DollRoom",
    "FakeMerchant",
    "PotionCourier",
    "RanwidTheElder",
    "RelicTrader",
    "RoomFullOfCheese",
    "SelfHelpBook",
    "SlipperyBridge",
    "StoneOfAllTime",
    "Symbiote",
    "TeaMaster",
    "TheFutureOfPotions",
    "TheLegendsWereTrue",
    "ThisOrThat",
    "WarHistorianRepy",
    "WelcomeToWongos",
]

# Ancient events never appear from Unknown "?" rooms (PullNextEvent only).
ANCIENT_EVENT_IDS: frozenset[str] = frozenset({
    "Neow",
    "Orobas",
    "Pael",
    "Tezcatara",
    "Nonupeipe",
    "Tanx",
    "Vakuu",
    "Darv",
})

SHARED_ANCIENT_IDS: list[str] = ["Darv"]

Act1BiomeChoice = Literal["random", "overgrowth", "underdocks"]
BiomeId = Literal["overgrowth", "underdocks", "hive", "glory"]


@dataclass
class ActConfig:
    """Configuration for a single act."""

    act_index: int
    num_rooms: int  # Number of room rows (used as mapLength input)
    biome_id: BiomeId = "overgrowth"
    num_weak_encounters: int = 3  # C# NumberOfWeakEncounters (3 for Act 1 biomes, 2 for 2/3)
    boss_ids: list[str] = field(default_factory=list)
    boss_id: str | None = None
    elite_ids: list[str] = field(default_factory=list)
    weak_encounter_ids: list[str] = field(default_factory=list)
    strong_encounter_ids: list[str] = field(default_factory=list)
    act_event_ids: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    ancient_ids: list[str] = field(default_factory=list)
    ancient_id: str | None = None
    events_visited: int = 0

    def to_mutable(self) -> "ActConfig":
        return ActConfig(
            act_index=self.act_index,
            num_rooms=self.num_rooms,
            biome_id=self.biome_id,
            num_weak_encounters=self.num_weak_encounters,
            boss_ids=list(self.boss_ids),
            boss_id=self.boss_id,
            elite_ids=list(self.elite_ids),
            weak_encounter_ids=list(self.weak_encounter_ids),
            strong_encounter_ids=list(self.strong_encounter_ids),
            act_event_ids=list(self.act_event_ids),
            event_ids=list(self.event_ids),
            ancient_ids=list(self.ancient_ids),
            ancient_id=self.ancient_id,
            events_visited=self.events_visited,
        )


# ── Act 1 biomes (act_index 0) ────────────────────────────────────────

ACT_1_OVERGROWTH = ActConfig(
    act_index=0,
    biome_id="overgrowth",
    num_rooms=15,
    boss_ids=["VantomBoss", "CeremonialBeastBoss", "TheKinBoss"],
    elite_ids=[
        "BygoneEffigyElite",
        "ByrdonisElite",
        "PhrogParasiteElite",
    ],
    weak_encounter_ids=[
        "NibbitsWeak",
        "SlimesWeak",
        "ShrinkerBeetleWeak",
        "FuzzyWurmCrawlerWeak",
    ],
    strong_encounter_ids=[
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
    act_event_ids=[
        "AromaOfChaos",
        "ByrdonisNest",
        "DenseVegetation",
        "JungleMazeAdventure",
        "LuminousChoir",
        "MorphicGrove",
        "SapphireSeed",
        "SunkenStatue",
        "TabletOfTruth",
        "UnrestSite",
        "Wellspring",
        "WhisperingHollow",
        "WoodCarvings",
    ],
    ancient_ids=["Neow"],
)

ACT_1_UNDERDOCKS = ActConfig(
    act_index=0,
    biome_id="underdocks",
    num_rooms=15,
    boss_ids=["WaterfallGiantBoss", "SoulFyshBoss", "LagavulinMatriarchBoss"],
    elite_ids=[
        "PhantasmalGardenersElite",
        "SkulkingColonyElite",
        "TerrorEelElite",
    ],
    weak_encounter_ids=[
        "CorpseSlugsWeak",
        "SeapunkWeak",
        "SludgeSpinnerWeak",
        "ToadpolesWeak",
    ],
    strong_encounter_ids=[
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
    act_event_ids=[
        "AbyssalBaths",
        "DrowningBeacon",
        "EndlessConveyor",
        "PunchOff",
        "SpiralingWhirlpool",
        "SunkenStatue",
        "SunkenTreasury",
        "DoorsOfLightAndDark",
        "TrashHeap",
        "WaterloggedScriptorium",
    ],
    ancient_ids=["Neow"],
)

# Backward-compatible alias (default Overgrowth-only run template).
ACT_0 = ACT_1_OVERGROWTH

# ── Acts 2 and 3 (Hive, Glory) ────────────────────────────────────────

ACT_1 = ActConfig(
    act_index=1,
    biome_id="hive",
    num_rooms=14,  # C# Hive.BaseNumberOfRooms = 14
    num_weak_encounters=2,  # C# Hive.NumberOfWeakEncounters = 2
    boss_ids=["TheInsatiableBoss", "KnowledgeDemonBoss", "KaiserCrabBoss"],
    elite_ids=[
        "DecimillipedeElite",
        "EntomancerElite",
        "InfestedPrismsElite",
    ],
    weak_encounter_ids=[
        "BowlbugsWeak",
        "ExoskeletonsWeak",
        "ThievingHopperWeak",
        "TunnelerWeak",
    ],
    strong_encounter_ids=[
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
    act_event_ids=[
        "Amalgamator",
        "Bugslayer",
        "ColorfulPhilosophers",
        "ColossalFlower",
        "FieldOfManSizedHoles",
        "InfestedAutomaton",
        "LostWisp",
        "SpiritGrafter",
        "TheLanternKey",
        "ZenWeaver",
    ],
    ancient_ids=["Orobas", "Pael", "Tezcatara"],
)

ACT_2 = ActConfig(
    act_index=2,
    biome_id="glory",
    num_rooms=13,  # C# Glory.BaseNumberOfRooms = 13
    num_weak_encounters=2,  # C# Glory.NumberOfWeakEncounters = 2
    boss_ids=["QueenBoss", "TestSubjectBoss", "DoormakerBoss"],
    elite_ids=[
        "KnightsElite",
        "MechaKnightElite",
        "SoulNexusElite",
    ],
    weak_encounter_ids=[
        "DevotedSculptorWeak",
        "ScrollsOfBitingWeak",
        "TurretOperatorWeak",
    ],
    strong_encounter_ids=[
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
    act_event_ids=[
        "BattlewornDummy",
        "GraveOfTheForgotten",
        "HungryForMushrooms",
        "Reflections",
        "RoundTeaParty",
        "Trial",
        "TinkerTime",
    ],
    ancient_ids=["Nonupeipe", "Tanx", "Vakuu"],
)

ALL_ACTS = [ACT_1_OVERGROWTH, ACT_1, ACT_2]


def _pick_act1_biome(
    rng: Rng,
    *,
    underdocks_unlocked: bool,
    underdocks_discovered: bool,
    act1_override: Act1BiomeChoice | None,
) -> BiomeId:
    if act1_override == "overgrowth":
        return "overgrowth"
    if act1_override == "underdocks":
        if not underdocks_unlocked:
            raise ValueError("Underdocks selected but underdocks_unlocked is False")
        return "underdocks"
    if not underdocks_unlocked:
        return "overgrowth"
    # First discovery forces Underdocks (singleplayer save progress).
    if not underdocks_discovered:
        return "underdocks"
    return "underdocks" if rng.next_bool() else "overgrowth"


def build_run_acts(
    rng: Rng,
    *,
    underdocks_unlocked: bool = True,
    underdocks_discovered: bool = True,
    act1_biome: Act1BiomeChoice = "random",
) -> list[ActConfig]:
    """Build the 3-act list for a run (mirrors ActModel.GetRandomList + lobby override)."""
    act1 = _pick_act1_biome(
        rng,
        underdocks_unlocked=underdocks_unlocked,
        underdocks_discovered=underdocks_discovered,
        act1_override=act1_biome if act1_biome != "random" else None,
    )
    act1_config = ACT_1_UNDERDOCKS if act1 == "underdocks" else ACT_1_OVERGROWTH
    return [act1_config.to_mutable(), ACT_1.to_mutable(), ACT_2.to_mutable()]


def get_act_config(act_index: int) -> ActConfig:
    if 0 <= act_index < len(ALL_ACTS):
        return ALL_ACTS[act_index]
    raise ValueError(f"Invalid act index: {act_index}")


def build_act_event_pool(act: ActConfig) -> list[str]:
    """Per-act regular event pool (act events + shared), excluding ancients."""
    return list(act.act_event_ids) + list(SHARED_EVENT_IDS)

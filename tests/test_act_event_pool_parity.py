"""Act and shared event pool parity vs decompiled act models and ModelDb."""

from sts2_env.map.acts import (
    ACT_1,
    ACT_1_OVERGROWTH,
    ACT_1_UNDERDOCKS,
    ACT_2,
    ANCIENT_EVENT_IDS,
    SHARED_ANCIENT_IDS,
    SHARED_EVENT_IDS,
)

# MegaCrit.Sts2.Core.Models.Acts.Overgrowth.AllEvents
_OVERGROWTH_ACT_EVENTS = [
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
]

# MegaCrit.Sts2.Core.Models.Acts.Underdocks.AllEvents
_UNDERDOCKS_ACT_EVENTS = [
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
]

# MegaCrit.Sts2.Core.Models.Acts.Hive.AllEvents
_HIVE_ACT_EVENTS = [
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
]

# MegaCrit.Sts2.Core.Models.Acts.Glory.AllEvents
_GLORY_ACT_EVENTS = [
    "BattlewornDummy",
    "GraveOfTheForgotten",
    "HungryForMushrooms",
    "Reflections",
    "RoundTeaParty",
    "Trial",
    "TinkerTime",
]

# MegaCrit.Sts2.Core.Models.ModelDb.AllSharedEvents
_ALL_SHARED_EVENTS = [
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

_ALL_ANCIENTS = {
    "Neow",
    "Orobas",
    "Pael",
    "Tezcatara",
    "Nonupeipe",
    "Tanx",
    "Vakuu",
    "Darv",
}


def test_shared_event_ids_match_model_db():
    assert SHARED_EVENT_IDS == _ALL_SHARED_EVENTS


def test_shared_ancient_ids_match_model_db():
    assert SHARED_ANCIENT_IDS == ["Darv"]


def test_ancient_event_ids_match_all_ancients_union():
    assert ANCIENT_EVENT_IDS == _ALL_ANCIENTS


def test_overgrowth_act_event_ids_match_decompiled():
    assert ACT_1_OVERGROWTH.act_event_ids == _OVERGROWTH_ACT_EVENTS


def test_underdocks_act_event_ids_match_decompiled():
    assert ACT_1_UNDERDOCKS.act_event_ids == _UNDERDOCKS_ACT_EVENTS


def test_hive_act_event_ids_match_decompiled():
    assert ACT_1.act_event_ids == _HIVE_ACT_EVENTS


def test_glory_act_event_ids_match_decompiled():
    assert ACT_2.act_event_ids == _GLORY_ACT_EVENTS

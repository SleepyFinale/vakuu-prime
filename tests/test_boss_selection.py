"""Tests for pre-rolled boss selection and map labels."""

from __future__ import annotations

from sts2_env.core.enums import MapPointType, RoomType
from sts2_env.encounters.registry import BOSS_SETUP_BY_ID, get_boss_setup
from sts2_env.map.labels import boss_display_name, map_point_label
from sts2_env.run.run_manager import RunManager
from sts2_env.run.run_state import RunState


def test_generate_rooms_assigns_boss_id_per_act():
    state = RunState(seed=42, character_id="Ironclad")
    state.initialize_run()
    for act in state.acts:
        assert act.boss_id is not None
        assert act.boss_id in act.boss_ids


def test_generate_rooms_boss_id_deterministic_for_seed():
    state_a = RunState(seed=99, character_id="Ironclad")
    state_a.initialize_run()
    state_b = RunState(seed=99, character_id="Ironclad")
    state_b.initialize_run()
    assert [act.boss_id for act in state_a.acts] == [act.boss_id for act in state_b.acts]


def test_map_point_label_boss_uses_boss_name_not_generic():
    state = RunState(seed=7, character_id="Ironclad")
    state.initialize_run()
    act = state.current_act
    label = map_point_label(MapPointType.BOSS, act)
    assert label != "Boss"
    assert label == boss_display_name(act.boss_id)


def test_boss_registry_covers_all_act_boss_ids():
    state = RunState(seed=0, character_id="Ironclad")
    state.initialize_run()
    for act in state.acts:
        for boss_id in act.boss_ids:
            assert boss_id in BOSS_SETUP_BY_ID
            assert get_boss_setup(boss_id) is not None


def test_boss_combat_uses_prerolled_encounter():
    mgr = RunManager(seed=123, character_id="Ironclad")
    act = mgr.run_state.current_act
    boss_id = act.boss_id
    assert boss_id is not None

    mgr._enter_combat(RoomType.BOSS)
    combat = mgr.get_combat_state()
    assert combat is not None

    if boss_id == "VantomBoss":
        assert len(combat.enemies) == 1
        assert combat.enemies[0].monster_id == "VANTOM"
    elif boss_id == "CeremonialBeastBoss":
        assert len(combat.enemies) == 1
        assert combat.enemies[0].monster_id == "CEREMONIAL_BEAST"
    elif boss_id == "TheKinBoss":
        assert [enemy.monster_id for enemy in combat.enemies] == [
            "KIN_FOLLOWER",
            "KIN_FOLLOWER",
            "KIN_PRIEST",
        ]
    else:
        raise AssertionError(f"Unexpected act 0 boss_id: {boss_id}")

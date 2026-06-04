"""Parity tests for Unknown (?) room resolution and event routing."""

from __future__ import annotations

from sts2_env.core.enums import MapPointType, RoomType
from sts2_env.map.acts import ANCIENT_EVENT_IDS, SHARED_EVENT_IDS, build_act_event_pool
from sts2_env.run.events import get_event, pick_event
from sts2_env.run.odds import UnknownMapPointOdds
from sts2_env.run.run_manager import RunManager
from sts2_env.run.run_state import RunState, UNLOCK_STATE_NUMBER_OF_RUNS_KEY
from sts2_env.core.rng import Rng


def test_act_event_pools_exclude_ancients():
    from sts2_env.map.acts import ALL_ACTS

    for act in ALL_ACTS:
        pool = build_act_event_pool(act)
        assert not ANCIENT_EVENT_IDS.intersection(pool)
        for ancient_id in act.ancient_ids:
            assert ancient_id in ANCIENT_EVENT_IDS


def test_unknown_event_pick_never_returns_ancient():
    run_state = RunState(seed=99)
    run_state.initialize_run()
    run_state.player.unlock_state[UNLOCK_STATE_NUMBER_OF_RUNS_KEY] = 1
    run_state.current_act_index = 2

    for _ in range(50):
        event = pick_event(run_state)
        if event is not None:
            assert event.event_id not in ANCIENT_EVENT_IDS


def test_ancient_map_point_uses_rolled_ancient_not_pick_event():
    run_state = RunState(seed=101)
    run_state.initialize_run()
    run_state.current_act_index = 2
    act = run_state.current_act
    assert act.ancient_id is not None
    assert act.ancient_id in act.ancient_ids

    ancient_event = get_event(act.ancient_id)
    assert ancient_event is not None
    events_visited_before = act.events_visited

    mgr = RunManager.__new__(RunManager)
    mgr._run_state = run_state
    mgr._enter_event(MapPointType.ANCIENT)

    assert mgr._event_model is ancient_event
    assert act.events_visited == events_visited_before


def test_unknown_room_pity_increases_monster_odds():
    odds = UnknownMapPointOdds()
    rs = RunState(seed=7)
    rs.initialize_run()
    rs.player.unlock_state[UNLOCK_STATE_NUMBER_OF_RUNS_KEY] = 1
    rng = Rng(7)
    initial = odds._current[RoomType.MONSTER]

    result = odds.roll(rng, rs)
    if result != RoomType.MONSTER:
        assert odds._current[RoomType.MONSTER] > initial
    else:
        assert odds._current[RoomType.MONSTER] == initial


def test_run_manager_unknown_event_is_not_ancient():
    mgr = RunManager(seed=202, character_id="Ironclad")
    mgr._run_state.player.unlock_state[UNLOCK_STATE_NUMBER_OF_RUNS_KEY] = 1
    mgr._run_state.current_act_index = 2

    act = mgr._run_state.current_act
    act.event_ids = [event_id for event_id in act.event_ids if event_id == "Trial"]
    act.events_visited = 0
    mgr._run_state.unknown_odds._current[RoomType.MONSTER] = -1.0
    mgr._run_state.unknown_odds._current[RoomType.TREASURE] = -1.0
    mgr._run_state.unknown_odds._current[RoomType.SHOP] = -1.0

    mgr._enter_room(RoomType.EVENT, MapPointType.UNKNOWN)

    assert mgr._event_model is not None
    assert mgr._event_model.event_id not in ANCIENT_EVENT_IDS
    assert mgr._event_model.event_id in act.event_ids or mgr._event_model.event_id in SHARED_EVENT_IDS

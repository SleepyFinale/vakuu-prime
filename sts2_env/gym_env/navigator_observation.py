"""Navigator-specific observation encoding for strategic run decisions.

Navigator obs v2 — 166 dims:
  0-14    run context (extended)
  15-22   phase one-hot (8)
  23-77   map branch options (5 x 11)
  78-82   path topology (5)
  83-127  card draft offers (5 x 9, combat-aligned)
  128-149 shop (22; last 2 = gold / max shop price)
  150-164 phase-specific options (15)
  165     deck value scalar
"""

from __future__ import annotations

from collections import deque

import numpy as np

from sts2_env.core.enums import MapPointType, RoomType
from sts2_env.gym_env.card_value import (
    MAX_CARD_OPTIONS,
    RUN_CONTEXT_SIZE,
    encode_run_context,
)
from sts2_env.gym_env.observation import (
    CARD_FEATURES as COMBAT_CARD_FEATURES,
    encode_card_features as encode_combat_card_features,
)
from sts2_env.gym_env.run_env import (
    NUM_PHASES,
    OBS_ACT_FLOOR_SCALE,
    OBS_ASCENSION_SCALE,
    OBS_CURRENT_ACT_SCALE,
    OBS_GOLD_SCALE,
    OBS_TOTAL_FLOOR_SCALE,
    OBS_VALUE_HIGH,
    OBS_VALUE_LOW,
    _PHASE_INDEX,
)
from sts2_env.map.map_point import MapCoord
from sts2_env.run.run_manager import RunManager
from sts2_env.run.run_state import RunState

MAX_MAP_OPTIONS = 5
NUM_MAP_POINT_TYPES = 9
MAP_OPTION_SIZE = NUM_MAP_POINT_TYPES + 2
RUN_CONTEXT_NAV_SIZE = 15
PATH_TOPOLOGY_SIZE = 5
SHOP_FEATURE_SIZE = 22
PHASE_OPTION_SIZE = 15
DECK_VALUE_FEATURE_SIZE = 1

_PATH_ELITE_NORM = 4.0
_PATH_REST_NORM = 4.0
_PATH_SHOP_NORM = 2.0

NAVIGATOR_OBS_SIZE = (
    RUN_CONTEXT_NAV_SIZE
    + NUM_PHASES
    + MAX_MAP_OPTIONS * MAP_OPTION_SIZE
    + PATH_TOPOLOGY_SIZE
    + MAX_CARD_OPTIONS * COMBAT_CARD_FEATURES
    + SHOP_FEATURE_SIZE
    + PHASE_OPTION_SIZE
    + DECK_VALUE_FEATURE_SIZE
)

_MAP_POINT_TYPES_ORDER = (
    MapPointType.UNASSIGNED,
    MapPointType.UNKNOWN,
    MapPointType.MONSTER,
    MapPointType.ELITE,
    MapPointType.BOSS,
    MapPointType.SHOP,
    MapPointType.REST_SITE,
    MapPointType.TREASURE,
    MapPointType.ANCIENT,
)


def _one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        vec[index] = 1.0
    return vec


def _encode_extended_run_context(mgr: RunManager) -> np.ndarray:
    """Run context with extra fields beyond card-value encoding."""
    base = encode_run_context(mgr)
    rs = mgr.run_state
    player = rs.player
    extended = np.zeros(RUN_CONTEXT_NAV_SIZE, dtype=np.float32)
    n = min(len(base), RUN_CONTEXT_NAV_SIZE)
    extended[:n] = base[:n]
    if RUN_CONTEXT_NAV_SIZE > RUN_CONTEXT_SIZE:
        extended[RUN_CONTEXT_SIZE] = rs.act_floor / OBS_ACT_FLOOR_SCALE
        room = mgr._current_room_type
        extended[RUN_CONTEXT_SIZE + 1] = 1.0 if room == RoomType.ELITE else 0.0
        extended[RUN_CONTEXT_SIZE + 2] = 1.0 if room == RoomType.BOSS else 0.0
        extended[RUN_CONTEXT_SIZE + 3] = rs.ascension_level / 20.0
        extended[RUN_CONTEXT_SIZE + 4] = player.current_hp / max(player.max_hp, 1)
    return extended


def _encode_map_point_type(point_type: MapPointType) -> np.ndarray:
    index = next(
        (i for i, pt in enumerate(_MAP_POINT_TYPES_ORDER) if pt == point_type),
        -1,
    )
    return _one_hot(index, NUM_MAP_POINT_TYPES)


def _encode_map_options(mgr: RunManager) -> np.ndarray:
    """Encode up to five available map branches."""
    features = np.zeros(MAX_MAP_OPTIONS * MAP_OPTION_SIZE, dtype=np.float32)
    act_map = mgr.run_state.map
    for i, coord in enumerate(mgr._available_coords[:MAX_MAP_OPTIONS]):
        offset = i * MAP_OPTION_SIZE
        point = act_map.get_point(coord) if act_map else None
        point_type = point.point_type if point else MapPointType.UNKNOWN
        features[offset: offset + NUM_MAP_POINT_TYPES] = _encode_map_point_type(point_type)
        features[offset + NUM_MAP_POINT_TYPES] = coord.row / OBS_TOTAL_FLOOR_SCALE
        features[offset + NUM_MAP_POINT_TYPES + 1] = coord.col / 10.0
    return features


def _current_map_coord(rs: RunState) -> MapCoord | None:
    act_map = rs.map
    if act_map is None:
        return None
    if rs.visited_map_coords:
        return rs.visited_map_coords[-1]
    if act_map.start_point is not None:
        return act_map.start_point.coord
    return None


def _reachable_unvisited_coords(rs: RunState) -> set[MapCoord]:
    """BFS forward from current position; skip already-visited nodes."""
    act_map = rs.map
    if act_map is None:
        return set()

    visited = set(rs.visited_map_coords)
    start = _current_map_coord(rs)
    if start is None:
        return set()

    reachable: set[MapCoord] = set()
    queue: deque[MapCoord] = deque([start])
    seen: set[MapCoord] = {start}

    while queue:
        coord = queue.popleft()
        point = act_map.get_point(coord)
        if point is None:
            continue
        for child in point.children:
            child_coord = child.coord
            if child_coord in seen:
                continue
            seen.add(child_coord)
            if child_coord not in visited:
                reachable.add(child_coord)
            queue.append(child_coord)

    return reachable


def _encode_path_topology(mgr: RunManager) -> np.ndarray:
    """Global map context: distance to boss, remaining elites, reachable rests/shops."""
    features = np.zeros(PATH_TOPOLOGY_SIZE, dtype=np.float32)
    rs = mgr.run_state
    act_map = rs.map
    if act_map is None or act_map.boss_point is None:
        features[4] = rs.current_act_index / OBS_CURRENT_ACT_SCALE
        return features

    current = _current_map_coord(rs)
    boss_row = act_map.boss_point.row
    if current is not None:
        floors = max(0, boss_row - current.row)
        features[0] = floors / max(act_map.map_length, 1)

    reachable = _reachable_unvisited_coords(rs)
    elites = rests = shops = 0
    for coord in reachable:
        point = act_map.get_point(coord)
        if point is None:
            continue
        if point.point_type == MapPointType.ELITE:
            elites += 1
        elif point.point_type == MapPointType.REST_SITE:
            rests += 1
        elif point.point_type == MapPointType.SHOP:
            shops += 1

    features[1] = elites / _PATH_ELITE_NORM
    features[2] = rests / _PATH_REST_NORM
    features[3] = shops / _PATH_SHOP_NORM
    features[4] = rs.current_act_index / OBS_CURRENT_ACT_SCALE
    return features


def _encode_card_offers(mgr: RunManager) -> np.ndarray:
    features = np.zeros(MAX_CARD_OPTIONS * COMBAT_CARD_FEATURES, dtype=np.float32)
    for i, card in enumerate(mgr._offered_cards[:MAX_CARD_OPTIONS]):
        offset = i * COMBAT_CARD_FEATURES
        features[offset: offset + COMBAT_CARD_FEATURES] = encode_combat_card_features(card)
    return features


def _max_shop_price(mgr: RunManager) -> int:
    inv = mgr._shop_inventory
    if inv is None:
        return 0
    prices: list[int] = []
    for entry in inv.cards:
        prices.append(entry.price)
    for entry in inv.colorless_cards:
        prices.append(entry.price)
    for entry in inv.relics:
        prices.append(entry.price)
    for entry in inv.potions:
        prices.append(entry.price)
    if not inv.removal_used:
        prices.append(inv.removal_cost)
    return max(prices) if prices else 0


def _encode_shop_features(mgr: RunManager) -> np.ndarray:
    features = np.zeros(SHOP_FEATURE_SIZE, dtype=np.float32)
    if mgr.phase != RunManager.PHASE_SHOP:
        return features

    actions = [
        a for a in mgr.get_available_actions()
        if a.get("action") not in ("leave", "leave_shop")
    ]
    features[0] = min(len(actions), 9) / 9.0
    action_slots_end = SHOP_FEATURE_SIZE - 2
    for i, action in enumerate(actions[:9]):
        slot = 1 + i * 2
        if slot + 1 >= action_slots_end:
            break
        action_type = str(action.get("action", ""))
        features[slot] = 1.0 if action_type == "buy_card" else 0.0
        features[slot + 1] = 1.0 if action_type in ("buy_relic", "buy_potion", "remove_card") else 0.0

    player = mgr.run_state.player
    features[SHOP_FEATURE_SIZE - 2] = player.gold / OBS_GOLD_SCALE
    features[SHOP_FEATURE_SIZE - 1] = _max_shop_price(mgr) / OBS_GOLD_SCALE
    return features


def _encode_phase_options(mgr: RunManager) -> np.ndarray:
    """Coarse encoding for rest, event, and boss-relic option screens."""
    features = np.zeros(PHASE_OPTION_SIZE, dtype=np.float32)
    phase = mgr.phase
    if phase == RunManager.PHASE_REST_SITE:
        options = [a for a in mgr.get_available_actions() if a.get("action") == "rest_option"]
        features[0] = min(len(options), 5) / 5.0
        for i, option in enumerate(options[:5]):
            option_id = str(option.get("option_id", "")).upper()
            features[1 + i] = 1.0 if option_id == "HEAL" else 0.0
            features[6 + i] = 1.0 if option_id == "SMITH" else 0.0
    elif phase == RunManager.PHASE_EVENT:
        options = [a for a in mgr.get_available_actions() if a.get("action") == "event_choice"]
        features[0] = min(len(options), 4) / 4.0
        for i in range(min(len(options), 4)):
            features[1 + i] = 1.0
    elif phase == RunManager.PHASE_BOSS_RELIC:
        features[0] = min(len(mgr._boss_relics), 3) / 3.0
        for i in range(min(len(mgr._boss_relics), 3)):
            features[1 + i] = 1.0
    elif phase == RunManager.PHASE_CARD_REWARD:
        features[0] = min(len(mgr._offered_cards), MAX_CARD_OPTIONS) / MAX_CARD_OPTIONS
        features[1] = 1.0 if any(a.get("action") == "skip" for a in mgr.get_available_actions()) else 0.0
    return features


def encode_navigator_observation(
    mgr: RunManager | None,
    *,
    deck_value: float = 0.0,
) -> np.ndarray:
    """Encode run state for the Navigator policy."""
    obs = np.zeros(NAVIGATOR_OBS_SIZE, dtype=np.float32)
    if mgr is None:
        return obs

    idx = 0
    obs[idx: idx + RUN_CONTEXT_NAV_SIZE] = _encode_extended_run_context(mgr)
    idx += RUN_CONTEXT_NAV_SIZE

    phase_idx = _PHASE_INDEX.get(mgr.phase, 0)
    obs[idx + phase_idx] = 1.0
    idx += NUM_PHASES

    map_block = _encode_map_options(mgr)
    obs[idx: idx + len(map_block)] = map_block
    idx += len(map_block)

    path_block = _encode_path_topology(mgr)
    obs[idx: idx + PATH_TOPOLOGY_SIZE] = path_block
    idx += PATH_TOPOLOGY_SIZE

    card_block = _encode_card_offers(mgr)
    obs[idx: idx + len(card_block)] = card_block
    idx += len(card_block)

    shop_block = _encode_shop_features(mgr)
    obs[idx: idx + len(shop_block)] = shop_block
    idx += len(shop_block)

    phase_block = _encode_phase_options(mgr)
    obs[idx: idx + len(phase_block)] = phase_block
    idx += len(phase_block)

    obs[idx] = deck_value
    np.clip(obs, OBS_VALUE_LOW, OBS_VALUE_HIGH, out=obs)
    return obs

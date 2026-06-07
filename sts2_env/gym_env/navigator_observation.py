"""Navigator-specific observation encoding for strategic run decisions."""

from __future__ import annotations

import numpy as np

from sts2_env.core.enums import MapPointType, RoomType
from sts2_env.gym_env.card_value import (
    CARD_FEATURE_SIZE,
    MAX_CARD_OPTIONS,
    RUN_CONTEXT_SIZE,
    encode_card_features,
    encode_run_context,
)
from sts2_env.gym_env.run_env import (
    NUM_PHASES,
    OBS_ACT_FLOOR_SCALE,
    OBS_ASCENSION_SCALE,
    OBS_CURRENT_ACT_SCALE,
    OBS_DECK_SIZE_SCALE,
    OBS_GOLD_SCALE,
    OBS_MAX_POTION_SLOTS_SCALE,
    OBS_RELIC_COUNT_SCALE,
    OBS_TOTAL_FLOOR_SCALE,
    OBS_VALUE_HIGH,
    OBS_VALUE_LOW,
    _PHASE_INDEX,
)
from sts2_env.run.run_manager import RunManager

MAX_MAP_OPTIONS = 5
NUM_MAP_POINT_TYPES = 9
MAP_OPTION_SIZE = NUM_MAP_POINT_TYPES + 2
RUN_CONTEXT_NAV_SIZE = 15
SHOP_FEATURE_SIZE = 20
PHASE_OPTION_SIZE = 15
DECK_VALUE_FEATURE_SIZE = 1

NAVIGATOR_OBS_SIZE = (
    RUN_CONTEXT_NAV_SIZE
    + NUM_PHASES
    + MAX_MAP_OPTIONS * MAP_OPTION_SIZE
    + MAX_CARD_OPTIONS * CARD_FEATURE_SIZE
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
        extended[RUN_CONTEXT_SIZE + 3] = rs.ascension_level / OBS_ASCENSION_SCALE
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


def _encode_card_offers(mgr: RunManager) -> np.ndarray:
    features = np.zeros(MAX_CARD_OPTIONS * CARD_FEATURE_SIZE, dtype=np.float32)
    for i, card in enumerate(mgr._offered_cards[:MAX_CARD_OPTIONS]):
        features[i * CARD_FEATURE_SIZE: (i + 1) * CARD_FEATURE_SIZE] = encode_card_features(card)
    return features


def _encode_shop_features(mgr: RunManager) -> np.ndarray:
    features = np.zeros(SHOP_FEATURE_SIZE, dtype=np.float32)
    if mgr.phase != RunManager.PHASE_SHOP:
        return features
    actions = [
        a for a in mgr.get_available_actions()
        if a.get("action") not in ("leave",)
    ]
    features[0] = min(len(actions), 9) / 9.0
    for i, action in enumerate(actions[:9]):
        slot = 1 + i * 2
        if slot + 1 >= SHOP_FEATURE_SIZE:
            break
        action_type = str(action.get("action", ""))
        features[slot] = 1.0 if action_type == "buy_card" else 0.0
        features[slot + 1] = 1.0 if action_type in ("buy_relic", "buy_potion", "remove_card") else 0.0
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

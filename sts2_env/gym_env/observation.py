"""Observation space encoding.

Compact flat float32 vector (1985 dimensions, obs v11):
  Player state:       hp/max_hp, block/50, energy, max_energy,
                      ascension/20, turn_count/20                    (6)
  Player powers:      all PowerId values (268, amount/20)           (268)
  Hand (10 cards):    id, cost, dmg, block, is_attack, is_power,
                      has_exhaust, has_retain, hit_count              (90)
                      Empty slots: id=0, cost=-0.2 sentinel; real cards use max(0,cost)/5
  Pile summaries:     draw, discard, exhaust counts + pile memory
                      (31) + reserved (3)                            (37)
  Enemies (5 slots):  alive, hp%, block, intent_onehot(5),
                      intent_dmg/60, intent_hits/min(hits,10)/10,
                      all powers (268)                              (278 * 5 = 1390)
  Character mechanics: one-hot(5), stars, orb cap/count, orbs(3*2), osty(3) (17)
  Relics (30 slots):  relic_id_norm, rarity, enabled, is_used_up,
                      counter_norm                                   (5 * 30 = 150)
  Potions (9 slots):  potion_id_norm, rarity, can_use_in_combat      (3 * 9 = 27)
Total: 6 + 268 + 90 + 37 + 1390 + 17 + 150 + 27 = 1985
"""

from __future__ import annotations

import numpy as np

from sts2_env.characters.all import SUPPORTED_TRAINING_CHARACTERS
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import CardId, CardType, IntentType, OrbType, PotionRarity, PowerId, RelicRarity
from sts2_env.core.constants import MAX_HAND_SIZE, MAX_ENEMIES, MAX_POTION_SLOTS
import sts2_env.potions.all  # noqa: F401 -- register all potions for ID table
from sts2_env.potions.base import PotionInstance, all_potion_models, get_potion_model
from sts2_env.gym_env.pile_distribution import (
    PILE_FEATURES,
    PILE_MEMORY_FEATURES,
    cards_from_combat,
    encode_pile_summaries,
    projected_next_draw_count,
)
from sts2_env.orbs.base import OrbQueue
from sts2_env.relics.base import RelicId, RelicInstance
from sts2_env.monsters.intents import Intent

# Card IDs list for normalised encoding
CARD_IDS = list(CardId)
NUM_CARD_IDS = len(CARD_IDS)
_CARD_ID_TO_IDX = {cid: i for i, cid in enumerate(CARD_IDS)}

# All combat powers (sorted by name; excludes legacy aliases and GENERIC placeholder)
_EXCLUDED_POWER_IDS = frozenset({
    PowerId.GENERIC,
    PowerId.GRAPPLE_POWER,
    PowerId.MANGLE_POWER,
    PowerId.FREE_POWER,
})

COMBAT_POWERS: tuple[PowerId, ...] = tuple(
    sorted((p for p in PowerId if p not in _EXCLUDED_POWER_IDS), key=lambda p: p.name)
)
PLAYER_POWERS = COMBAT_POWERS
ENEMY_POWERS = COMBAT_POWERS
NUM_PLAYER_POWERS = len(PLAYER_POWERS)
NUM_ENEMY_POWERS = len(ENEMY_POWERS)
_POWER_ID_TO_PLAYER_IDX: dict[PowerId, int] = {
    pid: i for i, pid in enumerate(PLAYER_POWERS)
}
_POWER_ID_TO_ENEMY_IDX: dict[PowerId, int] = {
    pid: i for i, pid in enumerate(ENEMY_POWERS)
}

# Intent types for one-hot (5)
INTENT_TYPES = [
    IntentType.ATTACK, IntentType.MULTI_ATTACK, IntentType.DEFEND,
    IntentType.BUFF, IntentType.DEBUFF,
]
NUM_INTENT_TYPES = len(INTENT_TYPES)
INTENT_DAMAGE_SCALE = 60.0
INTENT_HITS_CAP = 10
_TRACKED_INTENT_TYPES = frozenset(INTENT_TYPES)
_ATTACK_INTENT_TYPES = frozenset((IntentType.ATTACK, IntentType.MULTI_ATTACK))


def fold_move_intents(intents: list[Intent]) -> tuple[int, int, set[IntentType]]:
    """Aggregate multi-step move intents into total damage, hits, and tracked types."""
    intent_types: set[IntentType] = set()
    total_damage = 0
    total_hits = 0
    for intent in intents:
        if intent.intent_type in _TRACKED_INTENT_TYPES:
            intent_types.add(intent.intent_type)
        if intent.intent_type in _ATTACK_INTENT_TYPES:
            total_damage += intent.damage * intent.hits
            total_hits += intent.hits
    return total_damage, total_hits, intent_types


def intent_types_from_names(names: list[str]) -> set[IntentType]:
    """Map bridge intent name strings to tracked IntentType values."""
    intent_types: set[IntentType] = set()
    for name in names:
        try:
            intent_type = IntentType[str(name)]
        except KeyError:
            continue
        if intent_type in _TRACKED_INTENT_TYPES:
            intent_types.add(intent_type)
    return intent_types


def write_enemy_intent_features(
    obs: np.ndarray,
    base_idx: int,
    *,
    intent_types: set[IntentType],
    total_damage: int = 0,
    total_hits: int = 0,
) -> None:
    """Write folded intent one-hot bits plus scaled damage/hits into obs."""
    for intent_type in intent_types:
        for j, tracked_type in enumerate(INTENT_TYPES):
            if intent_type == tracked_type:
                obs[base_idx + j] = 1.0
    obs[base_idx + NUM_INTENT_TYPES] = total_damage / INTENT_DAMAGE_SCALE
    obs[base_idx + NUM_INTENT_TYPES + 1] = min(total_hits, INTENT_HITS_CAP) / INTENT_HITS_CAP


# Player core state (before powers)
PLAYER_CORE_FEATURES = 6  # hp, block, energy, max_energy, ascension, turn_count
OBS_ASCENSION_SCALE = 20.0
OBS_TURN_COUNT_CAP = 20.0

# Per-card features in hand
CARD_FEATURES = 9  # id, cost, dmg, block, is_attack, is_power, exhaust, retain, hits
EMPTY_HAND_SLOT_COST = -0.2  # sentinel for empty slots; real costs are in [0, 1]

# Per-enemy features
# alive(1) + hp%(1) + block(1) + intent_onehot(5) + intent_dmg(1) + intent_hits(1) + powers(268)
ENEMY_CORE_FEATURES = 1 + 1 + 1 + NUM_INTENT_TYPES + 1 + 1  # = 10
ENEMY_FEATURES = ENEMY_CORE_FEATURES + NUM_ENEMY_POWERS

# Pile summary features (counts + draw-pile memory + reserved padding)
# Exported from pile_distribution.py: PILE_FEATURES, PILE_MEMORY_FEATURES

# Character mechanics (orbs, stars, Osty companion)
NUM_TRAINING_CHARACTERS = len(SUPPORTED_TRAINING_CHARACTERS)
CHARACTER_ONE_HOT_FEATURES = NUM_TRAINING_CHARACTERS
STARS_FEATURES = 1
ORB_SUMMARY_FEATURES = 2  # capacity, count
ORB_SLOT_FEATURES = 3 * 2  # type_index, evoke_value per slot (first 3 slots)
OSTY_FEATURES = 3  # alive, hp_ratio, block
CHARACTER_MECHANICS_FEATURES = (
    CHARACTER_ONE_HOT_FEATURES
    + STARS_FEATURES
    + ORB_SUMMARY_FEATURES
    + ORB_SLOT_FEATURES
    + OSTY_FEATURES
)  # = 17

_CHARACTER_ID_TO_ONE_HOT: dict[str, int] = {
    char_id: index for index, char_id in enumerate(SUPPORTED_TRAINING_CHARACTERS)
}

_ORB_TYPE_NAME_TO_INDEX: dict[str, int] = {
    orb_type.name: index for index, orb_type in enumerate(OrbType)
}

# Base combat observation (before character mechanics)
BASE_OBS_SIZE = (
    PLAYER_CORE_FEATURES               # player state
    + NUM_PLAYER_POWERS                # player powers (268)
    + MAX_HAND_SIZE * CARD_FEATURES    # hand cards (90)
    + PILE_FEATURES                    # pile summaries (37)
    + MAX_ENEMIES * ENEMY_FEATURES     # enemies (1390)
)

# Combat observation before relic slots (v2 size)
COMBAT_OBS_V2_SIZE = BASE_OBS_SIZE + CHARACTER_MECHANICS_FEATURES  # = 148

# Relic slots (obs v3)
MAX_RELIC_SLOTS = 30
RELIC_FEATURES = 5  # relic_id_norm, rarity, enabled, is_used_up, counter_norm
RELIC_COUNTER_NORM = 20.0
RELIC_IDS = list(RelicId)
NUM_RELIC_IDS = len(RELIC_IDS)
_RELIC_ID_TO_IDX = {rid: i for i, rid in enumerate(RELIC_IDS)}
NUM_RELIC_RARITIES = len(RelicRarity)
_RELIC_RARITY_TO_IDX = {rarity: i for i, rarity in enumerate(RelicRarity)}
RELIC_OBS_SIZE = MAX_RELIC_SLOTS * RELIC_FEATURES  # = 150

# Potion slots (obs v5)
MAX_POTION_OBS_SLOTS = MAX_POTION_SLOTS
POTION_FEATURES = 3  # potion_id_norm, rarity_norm, can_use_in_combat_flag
POTION_IDS = sorted(model.potion_id for model in all_potion_models())
NUM_POTION_IDS = len(POTION_IDS)
_POTION_ID_TO_IDX = {pid: i for i, pid in enumerate(POTION_IDS)}
NUM_POTION_RARITIES = len(PotionRarity)
_POTION_RARITY_TO_IDX = {rarity: i for i, rarity in enumerate(PotionRarity)}
POTION_OBS_SIZE = MAX_POTION_OBS_SLOTS * POTION_FEATURES  # = 27

# Full observation size (obs v11)
OBS_SIZE = COMBAT_OBS_V2_SIZE + RELIC_OBS_SIZE + POTION_OBS_SIZE  # = 1985

_RELIC_SLICE_END = COMBAT_OBS_V2_SIZE + RELIC_OBS_SIZE

# Token layout for attention feature extractor (start, end indices)
TOKEN_SLICES: dict[str, tuple[int, int]] = {
    "player": (0, PLAYER_CORE_FEATURES + NUM_PLAYER_POWERS),
    "hand": (
        PLAYER_CORE_FEATURES + NUM_PLAYER_POWERS,
        PLAYER_CORE_FEATURES + NUM_PLAYER_POWERS + MAX_HAND_SIZE * CARD_FEATURES,
    ),
    "piles": (
        PLAYER_CORE_FEATURES + NUM_PLAYER_POWERS + MAX_HAND_SIZE * CARD_FEATURES,
        PLAYER_CORE_FEATURES + NUM_PLAYER_POWERS + MAX_HAND_SIZE * CARD_FEATURES + PILE_FEATURES,
    ),
    "enemies": (
        PLAYER_CORE_FEATURES + NUM_PLAYER_POWERS + MAX_HAND_SIZE * CARD_FEATURES + PILE_FEATURES,
        BASE_OBS_SIZE,
    ),
    "mechanics": (BASE_OBS_SIZE, COMBAT_OBS_V2_SIZE),
    "relics": (COMBAT_OBS_V2_SIZE, _RELIC_SLICE_END),
    "potions": (_RELIC_SLICE_END, OBS_SIZE),
}


def _card_hit_count_from_effect_vars(effect_vars: dict[str, int]) -> int:
    """Static hit count from card effect_vars (repeat or hits keys)."""
    return max(1, effect_vars.get("repeat", effect_vars.get("hits", 1)))


def _card_hit_count(card: object) -> int:
    effect_vars = getattr(card, "effect_vars", None) or {}
    return _card_hit_count_from_effect_vars(effect_vars)


def _normalize_card_keywords(keywords: object) -> frozenset[str]:
    if not keywords:
        return frozenset()
    if isinstance(keywords, str):
        return frozenset({keywords.lower()})
    return frozenset(str(keyword).lower() for keyword in keywords)


def _card_type_from_name(card_type: str | None) -> CardType:
    if not card_type:
        return CardType.SKILL
    normalized = card_type.strip().upper()
    if normalized == CardType.ATTACK.name:
        return CardType.ATTACK
    if normalized == CardType.POWER.name:
        return CardType.POWER
    if normalized in {CardType.STATUS.name, CardType.CURSE.name}:
        return CardType[normalized]
    return CardType.SKILL


def _coerce_bridge_card_id(card_id: str | None) -> CardId | None:
    if not card_id:
        return None
    normalized = card_id.strip()
    candidates = {
        normalized,
        normalized.upper(),
        "".join(("_" + ch if ch.isupper() else ch) for ch in normalized).upper().lstrip("_"),
    }
    for candidate in candidates:
        if candidate in CardId.__members__:
            return CardId[candidate]
    return None


def write_empty_hand_slot(obs: np.ndarray, idx: int) -> None:
    """Mark an unused hand slot; card-id norm stays 0."""
    obs[idx + 1] = EMPTY_HAND_SLOT_COST


def encode_card_features(card: object) -> np.ndarray:
    """Encode a single hand card as a float32 feature vector."""
    features = np.zeros(CARD_FEATURES, dtype=np.float32)
    card_id = getattr(card, "card_id", None)
    features[0] = (_CARD_ID_TO_IDX.get(card_id, 0) + 1) / (NUM_CARD_IDS + 1)
    features[1] = max(0, getattr(card, "cost", 0)) / 5.0
    base_damage = getattr(card, "base_damage", None)
    base_block = getattr(card, "base_block", None)
    features[2] = (base_damage or 0) / 50.0
    features[3] = (base_block or 0) / 50.0
    features[4] = 1.0 if getattr(card, "is_attack", False) else 0.0
    features[5] = 1.0 if getattr(card, "is_power", False) else 0.0
    features[6] = 1.0 if getattr(card, "exhausts", False) else 0.0
    features[7] = 1.0 if getattr(card, "should_retain_this_turn", False) else 0.0
    features[8] = _card_hit_count(card) / 5.0
    return features


def encode_card_features_from_fields(
    *,
    card_id: str | None = None,
    cost: int = 0,
    card_type: str | CardType | None = None,
    base_damage: int | None = None,
    base_block: int | None = None,
    keywords: object = None,
    retain: bool = False,
    single_turn_retain: bool = False,
    hit_count: int | None = None,
    upgraded: bool = False,
) -> np.ndarray:
    """Encode hand card features from plain fields (simulator or bridge JSON)."""
    features = np.zeros(CARD_FEATURES, dtype=np.float32)

    card_enum = _coerce_bridge_card_id(card_id) if card_id else None
    if card_enum is not None:
        features[0] = (_CARD_ID_TO_IDX.get(card_enum, 0) + 1) / (NUM_CARD_IDS + 1)

    keyword_set = _normalize_card_keywords(keywords)
    resolved_type = (
        card_type
        if isinstance(card_type, CardType)
        else _card_type_from_name(str(card_type) if card_type is not None else None)
    )
    resolved_damage = base_damage
    resolved_block = base_block
    resolved_hit_count = hit_count

    needs_reference = (
        card_enum is not None
        and (
            resolved_damage is None
            or resolved_block is None
            or not keyword_set
            or resolved_hit_count is None
        )
    )
    if needs_reference:
        try:
            from sts2_env.cards.factory import create_reference_card

            preview = create_reference_card(card_enum, upgraded=upgraded, allow_generation=False)
            if resolved_damage is None:
                resolved_damage = preview.base_damage
            if resolved_block is None:
                resolved_block = preview.base_block
            if not keyword_set:
                keyword_set = frozenset(keyword.lower() for keyword in preview.keywords)
            resolved_type = preview.card_type
            if resolved_hit_count is None:
                resolved_hit_count = _card_hit_count(preview)
        except (KeyError, AttributeError, TypeError):
            if resolved_hit_count is None:
                resolved_hit_count = 1

    features[1] = max(0, cost) / 5.0
    features[2] = (resolved_damage or 0) / 50.0
    features[3] = (resolved_block or 0) / 50.0
    features[4] = 1.0 if resolved_type == CardType.ATTACK else 0.0
    features[5] = 1.0 if resolved_type == CardType.POWER else 0.0
    features[6] = 1.0 if "exhaust" in keyword_set else 0.0
    features[7] = 1.0 if retain or single_turn_retain or "retain" in keyword_set else 0.0
    features[8] = max(1, resolved_hit_count or 1) / 5.0
    return features


def encode_character_mechanics_from_fields(
    obs: np.ndarray,
    start_idx: int,
    *,
    character_id: str | None = None,
    stars: int = 0,
    orb_capacity: int = 0,
    orb_count: int = 0,
    orbs: list[tuple[str, int]] | None = None,
    osty_alive: bool = False,
    osty_hp: int = 0,
    osty_max_hp: int = 0,
    osty_block: int = 0,
) -> int:
    """Encode character mechanics from plain fields (simulator or bridge JSON)."""
    idx = start_idx

    if character_id is not None:
        char_index = _CHARACTER_ID_TO_ONE_HOT.get(character_id)
        if char_index is None:
            normalized = character_id.strip()
            for known_id, known_index in _CHARACTER_ID_TO_ONE_HOT.items():
                if known_id.lower() == normalized.lower():
                    char_index = known_index
                    break
        if char_index is not None:
            obs[idx + char_index] = 1.0
    idx += CHARACTER_ONE_HOT_FEATURES

    obs[idx] = stars / 30.0
    idx += STARS_FEATURES

    obs[idx] = orb_capacity / OrbQueue.MAX_CAPACITY
    obs[idx + 1] = orb_count / OrbQueue.MAX_CAPACITY
    orb_entries = orbs or []
    for slot in range(3):
        slot_idx = idx + ORB_SUMMARY_FEATURES + slot * 2
        if slot < len(orb_entries):
            orb_type_name, evoke_value = orb_entries[slot]
            type_index = _ORB_TYPE_NAME_TO_INDEX.get(orb_type_name.upper(), 0)
            obs[slot_idx] = (type_index + 1) / len(OrbType)
            obs[slot_idx + 1] = evoke_value / 50.0
    idx += ORB_SUMMARY_FEATURES + ORB_SLOT_FEATURES

    if osty_alive and osty_max_hp > 0:
        obs[idx] = 1.0
        obs[idx + 1] = osty_hp / osty_max_hp
        obs[idx + 2] = osty_block / 50.0
    idx += OSTY_FEATURES

    return idx


def encode_relic_features(relic: RelicInstance) -> np.ndarray:
    """Encode a single relic as a float32 feature vector."""
    features = np.zeros(RELIC_FEATURES, dtype=np.float32)
    features[0] = (_RELIC_ID_TO_IDX.get(relic.relic_id, 0) + 1) / (NUM_RELIC_IDS + 1)
    rarity_index = _RELIC_RARITY_TO_IDX.get(relic.rarity, 0)
    features[1] = rarity_index / max(NUM_RELIC_RARITIES - 1, 1)
    features[2] = 1.0 if relic.enabled else 0.0
    features[3] = 1.0 if relic.is_used_up else 0.0
    features[4] = relic.counter / RELIC_COUNTER_NORM
    return features


def encode_relic_features_from_fields(
    *,
    relic_id: str | None = None,
    rarity: str | int | None = None,
    enabled: bool = True,
    is_used_up: bool = False,
    counter: int = 0,
) -> np.ndarray:
    """Encode relic features from plain fields (simulator or bridge JSON)."""
    from sts2_env.relics.registry import coerce_relic_id

    features = np.zeros(RELIC_FEATURES, dtype=np.float32)
    if relic_id:
        try:
            relic_enum = coerce_relic_id(relic_id.strip())
            features[0] = (_RELIC_ID_TO_IDX.get(relic_enum, 0) + 1) / (NUM_RELIC_IDS + 1)
        except KeyError:
            pass
    if rarity is not None:
        if isinstance(rarity, int):
            rarity_index = rarity
        else:
            rarity_name = str(rarity).strip().upper()
            rarity_enum = RelicRarity.__members__.get(rarity_name)
            rarity_index = _RELIC_RARITY_TO_IDX.get(rarity_enum, 0) if rarity_enum else 0
        features[1] = rarity_index / max(NUM_RELIC_RARITIES - 1, 1)
    features[2] = 1.0 if enabled else 0.0
    features[3] = 1.0 if is_used_up else 0.0
    features[4] = counter / RELIC_COUNTER_NORM
    return features


def encode_relics_into_obs(
    obs: np.ndarray,
    start_idx: int,
    relics: list[RelicInstance] | list[dict[str, object]] | None,
) -> int:
    """Write up to MAX_RELIC_SLOTS relic feature blocks starting at start_idx."""
    idx = start_idx
    if relics:
        for relic in relics[:MAX_RELIC_SLOTS]:
            if isinstance(relic, RelicInstance):
                obs[idx:idx + RELIC_FEATURES] = encode_relic_features(relic)
            elif isinstance(relic, dict):
                obs[idx:idx + RELIC_FEATURES] = encode_relic_features_from_fields(
                    relic_id=str(relic.get("id", "")),
                    rarity=relic.get("rarity"),
                    enabled=bool(relic.get("enabled", True)),
                    is_used_up=bool(relic.get("used_up", relic.get("is_used_up", False))),
                    counter=int(relic.get("counter", 0)),
                )
            idx += RELIC_FEATURES
    return start_idx + MAX_RELIC_SLOTS * RELIC_FEATURES


def encode_potion_features(potion: PotionInstance) -> np.ndarray:
    """Encode a single potion as a float32 feature vector."""
    features = np.zeros(POTION_FEATURES, dtype=np.float32)
    features[0] = (_POTION_ID_TO_IDX.get(potion.potion_id, 0) + 1) / (NUM_POTION_IDS + 1)
    rarity_index = _POTION_RARITY_TO_IDX.get(potion.rarity, 0)
    features[1] = rarity_index / max(NUM_POTION_RARITIES - 1, 1)
    features[2] = 1.0 if potion.can_use_in_combat() else 0.0
    return features


def encode_potion_features_from_fields(
    *,
    potion_id: str | None = None,
    rarity: str | int | PotionRarity | None = None,
    can_use_in_combat: bool = True,
) -> np.ndarray:
    """Encode potion features from plain fields (simulator or bridge JSON)."""
    features = np.zeros(POTION_FEATURES, dtype=np.float32)
    if potion_id:
        normalized = potion_id.strip()
        features[0] = (_POTION_ID_TO_IDX.get(normalized, 0) + 1) / (NUM_POTION_IDS + 1)
        if rarity is None:
            model = get_potion_model(normalized)
            if model is not None:
                rarity = model.rarity
    if rarity is not None:
        if isinstance(rarity, PotionRarity):
            rarity_index = _POTION_RARITY_TO_IDX.get(rarity, 0)
        elif isinstance(rarity, int):
            rarity_index = rarity
        else:
            rarity_name = str(rarity).strip().upper()
            rarity_enum = PotionRarity.__members__.get(rarity_name)
            rarity_index = _POTION_RARITY_TO_IDX.get(rarity_enum, 0) if rarity_enum else 0
        features[1] = rarity_index / max(NUM_POTION_RARITIES - 1, 1)
    features[2] = 1.0 if can_use_in_combat else 0.0
    return features


def encode_potions_into_obs(
    obs: np.ndarray,
    start_idx: int,
    potions: list[PotionInstance | dict[str, object] | None] | None,
) -> int:
    """Write MAX_POTION_OBS_SLOTS potion feature blocks starting at start_idx."""
    potion_list = potions or []
    for slot in range(MAX_POTION_OBS_SLOTS):
        idx = start_idx + slot * POTION_FEATURES
        if slot >= len(potion_list):
            continue
        potion = potion_list[slot]
        if potion is None:
            continue
        if isinstance(potion, PotionInstance):
            obs[idx:idx + POTION_FEATURES] = encode_potion_features(potion)
        elif isinstance(potion, dict):
            usage = str(potion.get("usage", "")).upper()
            can_use_in_combat = usage != "AUTOMATIC"
            if "can_use" in potion and not potion.get("can_use", True):
                can_use_in_combat = False
            obs[idx:idx + POTION_FEATURES] = encode_potion_features_from_fields(
                potion_id=str(potion.get("id", "")),
                rarity=potion.get("rarity"),
                can_use_in_combat=can_use_in_combat,
            )
    return start_idx + POTION_OBS_SIZE


def _encode_character_mechanics(combat: CombatState, obs: np.ndarray, start_idx: int) -> int:
    """Encode character-specific mechanics into obs starting at start_idx."""
    orb_entries: list[tuple[str, int]] = []
    orb_queue = combat.orb_queue
    orb_capacity = 0
    orb_count = 0
    if isinstance(orb_queue, OrbQueue):
        orb_capacity = orb_queue.capacity
        orb_count = len(orb_queue.orbs)
        for orb in orb_queue.orbs[:3]:
            orb_entries.append((
                orb.orb_type.name,
                orb.get_evoke_value(combat),
            ))

    osty = combat.get_osty()
    return encode_character_mechanics_from_fields(
        obs,
        start_idx,
        character_id=combat.character_id,
        stars=combat.stars,
        orb_capacity=orb_capacity,
        orb_count=orb_count,
        orbs=orb_entries,
        osty_alive=osty is not None and osty.is_alive,
        osty_hp=osty.current_hp if osty is not None else 0,
        osty_max_hp=osty.max_hp if osty is not None else 0,
        osty_block=osty.block if osty is not None else 0,
    )


def encode_observation(combat: CombatState) -> np.ndarray:
    """Encode combat state as a compact flat float32 vector."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    idx = 0

    # --- Player state (6) ---
    obs[idx] = combat.player.current_hp / combat.player.max_hp if combat.player.max_hp > 0 else 0.0
    obs[idx + 1] = combat.player.block / 50.0
    obs[idx + 2] = combat.energy / 10.0
    obs[idx + 3] = combat.max_energy / 10.0
    obs[idx + 4] = combat.ascension_level / OBS_ASCENSION_SCALE
    obs[idx + 5] = min(combat.turn_count, OBS_TURN_COUNT_CAP) / OBS_TURN_COUNT_CAP
    idx += PLAYER_CORE_FEATURES

    # --- Player powers (268) ---
    for pid in PLAYER_POWERS:
        obs[idx] = combat.player.get_power_amount(pid) / 20.0
        idx += 1

    # --- Hand cards (10 * 9 = 90) ---
    for i in range(MAX_HAND_SIZE):
        if i < len(combat.hand):
            obs[idx:idx + CARD_FEATURES] = encode_card_features(combat.hand[i])
        else:
            write_empty_hand_slot(obs, idx)
        idx += CARD_FEATURES

    # --- Pile summaries (32) ---
    draw, discard, play, hand_cards = cards_from_combat(combat)
    next_draw_count = projected_next_draw_count(len(combat.hand), combat=combat)
    obs[idx:idx + PILE_FEATURES] = encode_pile_summaries(
        draw,
        discard,
        play,
        hand_cards,
        len(combat.exhaust_pile),
        next_draw_count=next_draw_count,
    )
    idx += PILE_FEATURES

    # --- Enemies (5 * 278 = 1390) ---
    for i in range(MAX_ENEMIES):
        if i < len(combat.enemies):
            enemy = combat.enemies[i]
            obs[idx] = 1.0 if enemy.is_alive else 0.0
            obs[idx + 1] = enemy.current_hp / enemy.max_hp if enemy.max_hp > 0 else 0.0
            obs[idx + 2] = enemy.block / 50.0

            # Intent encoding (multi-bit one-hot + aggregated damage + hits)
            ai = combat.enemy_ais.get(enemy.combat_id)
            if ai is not None and enemy.is_alive:
                intents = ai.current_move.intents
                if intents:
                    total_damage, total_hits, intent_types = fold_move_intents(intents)
                    write_enemy_intent_features(
                        obs,
                        idx + 3,
                        intent_types=intent_types,
                        total_damage=total_damage,
                        total_hits=total_hits,
                    )

            power_base = idx + ENEMY_CORE_FEATURES
            for j, pid in enumerate(ENEMY_POWERS):
                obs[power_base + j] = enemy.get_power_amount(pid) / 10.0
        idx += ENEMY_FEATURES  # advance even for empty enemy slots

    idx = _encode_character_mechanics(combat, obs, idx)
    idx = encode_relics_into_obs(obs, idx, combat.relics)
    encode_potions_into_obs(obs, idx, combat.potions)

    return obs

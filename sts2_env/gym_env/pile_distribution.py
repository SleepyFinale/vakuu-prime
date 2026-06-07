"""Draw-pile memory features for combat observations (obs v4).

Encodes unseen-deck composition, next-turn draw probabilities, known top-of-deck
order, and high-value card signals so feed-forward policies can count cards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from sts2_env.core.constants import BASE_DRAW, MAX_HAND_SIZE
from sts2_env.core.enums import CardId, CardRarity, CardTag, CardType

PILE_MEMORY_FEATURES = 26
PILE_COUNT_FEATURES = 3
PILE_RESERVED_FEATURES = 3
PILE_FEATURES = PILE_COUNT_FEATURES + PILE_MEMORY_FEATURES + PILE_RESERVED_FEATURES

_BRIDGE_CARD_TYPE_ATTACK = "Attack"
_BRIDGE_CARD_TYPE_SKILL = "Skill"
_BRIDGE_CARD_TYPE_POWER = "Power"
_BRIDGE_CARD_TYPE_STATUS = "Status"
_BRIDGE_CARD_TYPE_CURSE = "Curse"

_TYPE_ATTACK = 1.0
_TYPE_SKILL = 0.66
_TYPE_POWER = 0.33
_TYPE_OTHER = 0.0

_HEAVY_ATTACK_DAMAGE = 8
_RARE_PLUS = frozenset({CardRarity.RARE, CardRarity.ANCIENT})

WATCHLIST_GROUPS: dict[str, frozenset[CardId]] = {
    "power": frozenset({
        CardId.DEMON_FORM_CARD,
        CardId.CORRUPTION_CARD,
        CardId.WELL_LAID_PLANS,
        CardId.CREATIVE_AI_CARD,
        CardId.AFTERIMAGE_CARD,
        CardId.FEEL_NO_PAIN_CARD,
        CardId.STORM_CARD,
        CardId.CONSUMING_SHADOW,
    }),
    "finisher": frozenset({
        CardId.BASH,
        CardId.POMMEL_STRIKE,
        CardId.NEUTRALIZE,
        CardId.ADRENALINE,
        CardId.UNLEASH,
        CardId.DEATH_MARCH,
        CardId.SOVEREIGN_BLADE,
        CardId.GRAND_FINALE,
        CardId.ASSASSINATE,
        CardId.FIEND_FIRE,
    }),
    "setup": frozenset({
        CardId.FEEL_NO_PAIN_CARD,
        CardId.STORM_CARD,
        CardId.CLAW,
        CardId.CONSUMING_SHADOW,
        CardId.WELL_LAID_PLANS,
        CardId.BOOT_SEQUENCE,
        CardId.HIDDEN_CACHE,
    }),
    "aoe": frozenset({
        CardId.SOUL_STORM,
        CardId.BONE_SHARDS,
        CardId.STORM_OF_STEEL,
        CardId.FIEND_FIRE,
        CardId.DAGGER_SPRAY,
        CardId.CELESTIAL_MIGHT,
    }),
}


class PileCardView(Protocol):
    card_id: CardId
    card_type: CardType
    base_damage: int | None
    base_block: int | None
    cost: int
    tags: frozenset[CardTag]
    rarity: CardRarity
    should_retain: bool


@dataclass(frozen=True)
class SimplePileCard:
    card_id: CardId
    card_type: CardType
    base_damage: int | None = None
    base_block: int | None = None
    cost: int = 0
    tags: frozenset[CardTag] = frozenset()
    rarity: CardRarity = CardRarity.COMMON
    should_retain: bool = False

    @property
    def is_attack(self) -> bool:
        return self.card_type == CardType.ATTACK

    @property
    def is_skill(self) -> bool:
        return self.card_type == CardType.SKILL

    @property
    def is_power(self) -> bool:
        return self.card_type == CardType.POWER


def hypergeom_at_least_one(n: int, k: int, success_count: int) -> float:
    """P(draw >= 1 success in k draws without replacement from n cards)."""
    if k <= 0 or success_count <= 0 or n <= 0:
        return 0.0
    if success_count >= n:
        return 1.0
    if k >= n:
        return 1.0
    # P(none) = prod((n-K-i)/(n-i) for i in 0..k-1)
    fail_count = n - success_count
    if fail_count < k:
        return 1.0
    prob_none = 1.0
    for i in range(k):
        prob_none *= (fail_count - i) / (n - i)
    return float(max(0.0, min(1.0, 1.0 - prob_none)))


def hypergeom_expected_count(n: int, k: int, success_count: int) -> float:
    """Expected number of successes in k draws without replacement."""
    if k <= 0 or n <= 0 or success_count <= 0:
        return 0.0
    draw_count = min(k, n)
    return float(draw_count * success_count / n)


def _type_encoding(card_type: CardType) -> float:
    if card_type == CardType.ATTACK:
        return _TYPE_ATTACK
    if card_type == CardType.SKILL:
        return _TYPE_SKILL
    if card_type == CardType.POWER:
        return _TYPE_POWER
    return _TYPE_OTHER


def _card_type_from_name(type_name: str) -> CardType:
    mapping = {
        _BRIDGE_CARD_TYPE_ATTACK: CardType.ATTACK,
        _BRIDGE_CARD_TYPE_SKILL: CardType.SKILL,
        _BRIDGE_CARD_TYPE_POWER: CardType.POWER,
        _BRIDGE_CARD_TYPE_STATUS: CardType.STATUS,
        _BRIDGE_CARD_TYPE_CURSE: CardType.CURSE,
    }
    return mapping.get(type_name, CardType.SKILL)


def _coerce_card_id(card_id_str: str) -> CardId | None:
    if not card_id_str:
        return None
    normalized = card_id_str.strip()
    if normalized in CardId.__members__:
        return CardId[normalized]
    for member in CardId:
        if member.name.lower() == normalized.lower():
            return member
    candidates = {
        normalized,
        normalized.upper(),
        "".join(("_" + ch if ch.isupper() else ch) for ch in normalized).upper().lstrip("_"),
    }
    for candidate in candidates:
        if candidate in CardId.__members__:
            return CardId[candidate]
    aliases = {
        "Strike": CardId.STRIKE_IRONCLAD,
        "Defend": CardId.DEFEND_IRONCLAD,
        "Bash": CardId.BASH,
    }
    return aliases.get(normalized)


def _card_from_instance(card: object) -> SimplePileCard:
    return SimplePileCard(
        card_id=card.card_id,
        card_type=card.card_type,
        base_damage=card.base_damage,
        base_block=card.base_block,
        cost=max(0, getattr(card, "cost", 0)),
        tags=frozenset(getattr(card, "tags", frozenset())),
        rarity=getattr(card, "rarity", CardRarity.COMMON),
        should_retain=bool(getattr(card, "should_retain_this_turn", False)),
    )


def _card_from_bridge_dict(card: dict[str, Any]) -> SimplePileCard:
    card_id_str = str(card.get("id", ""))
    card_id = _coerce_card_id(card_id_str) or CardId.GENERIC
    card_type = _card_type_from_name(str(card.get("type", _BRIDGE_CARD_TYPE_SKILL)))
    base_damage = card.get("base_damage")
    base_block = card.get("base_block")
    cost = int(card.get("cost", 0))
    upgraded = bool(card.get("upgraded", False))
    tags: frozenset[CardTag] = frozenset()
    rarity = CardRarity.COMMON

    if base_damage is None or base_block is None or not tags:
        try:
            from sts2_env.cards.factory import create_reference_card

            preview = create_reference_card(card_id, upgraded=upgraded, allow_generation=False)
            if base_damage is None:
                base_damage = preview.base_damage
            if base_block is None:
                base_block = preview.base_block
            tags = frozenset(preview.tags)
            rarity = preview.rarity
            card_type = preview.card_type
        except (KeyError, AttributeError):
            pass

    retain_keywords = card.get("keywords") or []
    should_retain = bool(card.get("retain", False)) or "retain" in retain_keywords

    return SimplePileCard(
        card_id=card_id,
        card_type=card_type,
        base_damage=base_damage,
        base_block=base_block,
        cost=cost,
        tags=tags,
        rarity=rarity,
        should_retain=should_retain,
    )


def cards_from_combat(combat: object) -> tuple[list[SimplePileCard], list[SimplePileCard], list[SimplePileCard], list[SimplePileCard]]:
    """Return draw, discard, play, and hand card views from CombatState."""
    draw = [_card_from_instance(card) for card in combat.draw_pile]
    discard = [_card_from_instance(card) for card in combat.discard_pile]
    play = [_card_from_instance(card) for card in combat.play_pile]
    hand = [_card_from_instance(card) for card in combat.hand]
    return draw, discard, play, hand


def cards_from_bridge(combat_json: dict[str, Any]) -> tuple[list[SimplePileCard], list[SimplePileCard], list[SimplePileCard], list[SimplePileCard]]:
    """Return draw, discard, play, and hand card views from bridge JSON."""

    def _parse_cards(key: str) -> list[SimplePileCard]:
        cards: list[SimplePileCard] = []
        for raw in combat_json.get(key, []):
            if isinstance(raw, dict):
                cards.append(_card_from_bridge_dict(raw))
        return cards

    return (
        _parse_cards("draw_pile"),
        _parse_cards("discard_pile"),
        _parse_cards("play_pile"),
        _parse_cards("hand"),
    )


def projected_next_draw_count(
    hand_size: int,
    *,
    combat: object | None = None,
) -> int:
    """Estimate cards drawn at next turn start."""
    base_draw = BASE_DRAW
    if combat is not None:
        try:
            from sts2_env.core.hooks import modify_hand_draw

            owner = getattr(combat, "player", None)
            base_draw = modify_hand_draw(BASE_DRAW, combat, owner)
        except Exception:
            base_draw = BASE_DRAW
    hand_room = max(0, MAX_HAND_SIZE - hand_size)
    return max(0, min(base_draw, hand_room))


def _unseen_cards(
    draw: Sequence[PileCardView],
    discard: Sequence[PileCardView],
    play: Sequence[PileCardView],
) -> list[PileCardView]:
    return list(draw) + list(discard) + list(play)


def _type_fraction(cards: Sequence[PileCardView], card_type: CardType) -> float:
    if not cards:
        return 0.0
    return sum(1 for card in cards if card.card_type == card_type) / len(cards)


def _count_type(cards: Sequence[PileCardView], card_type: CardType) -> int:
    return sum(1 for card in cards if card.card_type == card_type)


def _prob_at_least_one_type(
    known_cards: Sequence[PileCardView],
    shuffle_pool: Sequence[PileCardView],
    *,
    remaining_draws: int,
    card_type: CardType,
) -> float:
    if any(card.card_type == card_type for card in known_cards):
        return 1.0
    if remaining_draws <= 0:
        return 0.0
    pool_size = len(shuffle_pool)
    success_count = _count_type(shuffle_pool, card_type)
    return hypergeom_at_least_one(pool_size, remaining_draws, success_count)


def _expected_type_draws(
    known_cards: Sequence[PileCardView],
    shuffle_pool: Sequence[PileCardView],
    *,
    remaining_draws: int,
    card_type: CardType,
) -> float:
    known = sum(1 for card in known_cards if card.card_type == card_type)
    pool_size = len(shuffle_pool)
    success_count = _count_type(shuffle_pool, card_type)
    expected = known + hypergeom_expected_count(pool_size, remaining_draws, success_count)
    return expected / float(BASE_DRAW)


def _heuristic_fraction(cards: Sequence[PileCardView], predicate) -> float:
    if not cards:
        return 0.0
    return sum(1 for card in cards if predicate(card)) / len(cards)


def _watchlist_presence(cards: Sequence[PileCardView], watchlist: frozenset[CardId]) -> float:
    return 1.0 if any(card.card_id in watchlist for card in cards) else 0.0


def encode_pile_memory(
    draw: Sequence[PileCardView],
    discard: Sequence[PileCardView],
    play: Sequence[PileCardView],
    hand: Sequence[PileCardView],
    *,
    next_draw_count: int | None = None,
) -> np.ndarray:
    """Encode draw-pile memory as a 26-dim float32 vector."""
    features = np.zeros(PILE_MEMORY_FEATURES, dtype=np.float32)
    if next_draw_count is None:
        next_draw_count = projected_next_draw_count(len(hand))

    unseen = _unseen_cards(draw, discard, play)
    idx = 0

    # Unseen deck composition (5)
    for card_type in (CardType.ATTACK, CardType.SKILL, CardType.POWER, CardType.STATUS, CardType.CURSE):
        features[idx] = _type_fraction(unseen, card_type)
        idx += 1

    pending_discard = list(play) + [card for card in hand if not card.should_retain]
    known_count = min(next_draw_count, len(draw))
    known_cards = list(draw[:known_count])
    remaining_draws = max(0, next_draw_count - known_count)
    shuffle_pool = list(draw[known_count:]) + list(discard) + pending_discard

    # Next-turn draw probabilities (5)
    for card_type in (CardType.ATTACK, CardType.SKILL, CardType.POWER):
        features[idx] = _prob_at_least_one_type(
            known_cards,
            shuffle_pool,
            remaining_draws=remaining_draws,
            card_type=card_type,
        )
        idx += 1
    features[idx] = _expected_type_draws(
        known_cards, shuffle_pool, remaining_draws=remaining_draws, card_type=CardType.ATTACK,
    )
    idx += 1
    features[idx] = _expected_type_draws(
        known_cards, shuffle_pool, remaining_draws=remaining_draws, card_type=CardType.SKILL,
    )
    idx += 1

    # Known draw order (5)
    for slot in range(5):
        if slot < len(draw):
            features[idx] = _type_encoding(draw[slot].card_type)
        idx += 1

    # Shuffle uncertainty (1)
    features[idx] = 1.0 if next_draw_count > len(draw) and len(shuffle_pool) > 0 else 0.0
    idx += 1

    # Heuristic high-value fractions (6)
    features[idx] = _heuristic_fraction(
        unseen,
        lambda card: card.is_attack and (card.base_damage or 0) >= _HEAVY_ATTACK_DAMAGE,
    )
    idx += 1
    features[idx] = _type_fraction(unseen, CardType.POWER)
    idx += 1
    features[idx] = _heuristic_fraction(unseen, lambda card: card.cost <= 0)
    idx += 1
    features[idx] = _heuristic_fraction(unseen, lambda card: card.rarity in _RARE_PLUS)
    idx += 1
    features[idx] = _heuristic_fraction(unseen, lambda card: CardTag.STRIKE in card.tags)
    idx += 1
    features[idx] = _heuristic_fraction(unseen, lambda card: CardTag.DEFEND in card.tags)
    idx += 1

    # Curated watchlist groups (4)
    for group_name in ("power", "finisher", "setup", "aoe"):
        features[idx] = _watchlist_presence(unseen, WATCHLIST_GROUPS[group_name])
        idx += 1

    return features


def encode_pile_summaries(
    draw: Sequence[PileCardView],
    discard: Sequence[PileCardView],
    play: Sequence[PileCardView],
    hand: Sequence[PileCardView],
    exhaust_size: int,
    *,
    next_draw_count: int | None = None,
) -> np.ndarray:
    """Encode full pile block: counts, memory, and reserved padding."""
    obs = np.zeros(PILE_FEATURES, dtype=np.float32)
    obs[0] = len(draw) / 20.0
    obs[1] = len(discard) / 20.0
    obs[2] = exhaust_size / 20.0
    obs[PILE_COUNT_FEATURES:PILE_COUNT_FEATURES + PILE_MEMORY_FEATURES] = encode_pile_memory(
        draw,
        discard,
        play,
        hand,
        next_draw_count=next_draw_count,
    )
    return obs

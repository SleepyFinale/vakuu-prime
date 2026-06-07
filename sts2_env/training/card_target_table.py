"""Static card target lookup for structural GNN edges."""

from __future__ import annotations

from functools import lru_cache

from sts2_env.core.enums import CardId, TargetType
from sts2_env.gym_env.observation import CARD_IDS, NUM_CARD_IDS


def card_id_index_from_norm(norm: float) -> int | None:
    """Invert card_id_norm = (idx + 1) / (NUM_CARD_IDS + 1)."""
    if norm <= 0.0:
        return None
    scaled = norm * (NUM_CARD_IDS + 1)
    index = int(round(scaled)) - 1
    if 0 <= index < NUM_CARD_IDS:
        return index
    return None


@lru_cache(maxsize=1)
def card_target_by_index() -> tuple[TargetType, ...]:
    """Return target type per CARD_IDS index (built once from static metadata)."""
    from sts2_env.cards.reference_static_metadata import reference_metadata_by_card_id

    metadata = reference_metadata_by_card_id()
    targets: list[TargetType] = []
    for card_id in CARD_IDS:
        entry = metadata.get(card_id)
        if entry is None:
            targets.append(TargetType.SELF)
        else:
            targets.append(entry.target_type)
    return tuple(targets)


def card_target_type_from_norm(norm: float) -> TargetType | None:
    """Resolve TargetType for a hand-card feature norm value."""
    index = card_id_index_from_norm(norm)
    if index is None:
        return None
    return card_target_by_index()[index]

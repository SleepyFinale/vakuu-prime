"""Map point display labels for UI."""

from __future__ import annotations

import re

from sts2_env.core.enums import MapPointType
from sts2_env.map.acts import ActConfig

ROOM_LABELS = {
    "MONSTER": "Monster",
    "ELITE": "Elite",
    "BOSS": "Boss",
    "SHOP": "Shop",
    "TREASURE": "Treasure",
    "REST_SITE": "Rest",
    "UNKNOWN": "?",
    "ANCIENT": "Event",
    "UNASSIGNED": "?",
}


def display_name(value: object) -> str:
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    text = text.replace("_", " ")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    return text.title()


def boss_display_name(boss_id: str) -> str:
    """Friendly boss name for map nodes (e.g. VantomBoss -> Vantom)."""
    name = display_name(boss_id)
    if name.endswith(" Boss"):
        return name[:-5]
    return name


def map_point_label(point_type: MapPointType, act: ActConfig) -> str:
    if point_type == MapPointType.BOSS and act.boss_id:
        return boss_display_name(act.boss_id)
    key = point_type.name if hasattr(point_type, "name") else str(point_type)
    return ROOM_LABELS.get(key, display_name(key))

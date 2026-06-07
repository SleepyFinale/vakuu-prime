"""Validate and regenerate docs/PILE_WATCHLIST.json from CardId metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.sync.common import DOCS_DIR

WATCHLIST_JSON_PATH = DOCS_DIR / "PILE_WATCHLIST.json"
WATCHLIST_GROUP_NAMES: tuple[str, ...] = ("power", "finisher", "setup", "aoe")

_RARITY_RANK: dict[str, int] = {
    "BASIC": 0,
    "COMMON": 1,
    "UNCOMMON": 2,
    "RARE": 3,
    "ANCIENT": 4,
    "STATUS": 5,
    "CURSE": 6,
    "EVENT": 7,
    "QUEST": 8,
}


@dataclass
class WatchlistSummary:
    added: dict[str, list[str]] = field(default_factory=dict)
    removed: list[str] = field(default_factory=list)
    unlisted_new: list[str] = field(default_factory=list)

    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.unlisted_new)

    def report_lines(self) -> list[str]:
        lines: list[str] = []
        for group_name in WATCHLIST_GROUP_NAMES:
            names = self.added.get(group_name, [])
            if names:
                lines.append(f"Auto-added to {group_name}: {', '.join(names)}")
        if self.removed:
            lines.append(f"Removed stale: {', '.join(self.removed)}")
        if self.unlisted_new:
            preview = ", ".join(self.unlisted_new[:20])
            suffix = f" ... and {len(self.unlisted_new) - 20} more" if len(self.unlisted_new) > 20 else ""
            lines.append(f"New cards not in any group (manual review): {preview}{suffix}")
        return lines


def default_watchlist_document() -> dict[str, Any]:
    from sts2_env.gym_env.pile_distribution import seed_watchlist_document

    return seed_watchlist_document()


def load_watchlist_document(path: Path = WATCHLIST_JSON_PATH) -> dict[str, Any]:
    if not path.is_file():
        return default_watchlist_document()
    return json.loads(path.read_text(encoding="utf-8"))


def _rarity_at_least(rarity_name: str, minimum_name: str) -> bool:
    return _RARITY_RANK.get(rarity_name, -1) >= _RARITY_RANK.get(minimum_name, 0)


def _matches_auto_rule(card: object, rule: dict[str, Any]) -> bool:
    from sts2_env.core.enums import CardRarity, CardType, TargetType

    card_type_name = rule.get("card_type")
    if card_type_name is not None and card.card_type != CardType[card_type_name]:
        return False

    target_type_name = rule.get("target_type")
    if target_type_name is not None and card.target_type != TargetType[target_type_name]:
        return False

    min_damage = rule.get("min_base_damage")
    if min_damage is not None and (card.base_damage or 0) < int(min_damage):
        return False

    rarity_min = rule.get("rarity_min")
    if rarity_min is not None and not _rarity_at_least(card.rarity.name, str(rarity_min)):
        return False

    return True


def _playable_card_ids() -> list:
    from sts2_env.cards.factory import card_preview
    from sts2_env.core.enums import CardId, CardType

    playable = []
    for card_id in CardId:
        try:
            card = card_preview(card_id)
        except (KeyError, AttributeError, ValueError):
            continue
        if card.card_type in (CardType.STATUS, CardType.CURSE):
            continue
        if not card.can_be_generated_in_combat and card.rarity.name in {"TOKEN"}:
            continue
        playable.append(card_id)
    return playable


def _coerce_card_name(name: str) -> str | None:
    from sts2_env.core.enums import CardId

    normalized = name.strip()
    if normalized in CardId.__members__:
        return normalized
    for member in CardId:
        if member.name.lower() == normalized.lower():
            return member.name
    return None


def generate_pile_watchlist(
    *,
    output: Path = WATCHLIST_JSON_PATH,
    apply_auto: bool = True,
) -> tuple[Path, WatchlistSummary]:
    """Validate watchlist JSON and auto-append new cards matching group rules."""
    from sts2_env.cards.factory import card_preview
    from sts2_env.core.enums import CardId

    document = load_watchlist_document(output)
    previous_known = {str(name) for name in document.get("known_card_ids", [])}
    groups: dict[str, Any] = document.setdefault("groups", {})
    summary = WatchlistSummary()

    for group_name in WATCHLIST_GROUP_NAMES:
        group = groups.setdefault(group_name, {"cards": [], "auto": None, "exclude": []})
        group.setdefault("exclude", [])

        valid_cards: list[str] = []
        for raw_name in group.get("cards", []):
            coerced = _coerce_card_name(str(raw_name))
            if coerced is None:
                summary.removed.append(str(raw_name))
            else:
                valid_cards.append(coerced)
        group["cards"] = sorted(set(valid_cards))

    all_listed: set[CardId] = set()
    for group_name in WATCHLIST_GROUP_NAMES:
        for raw_name in groups[group_name].get("cards", []):
            coerced = _coerce_card_name(str(raw_name))
            if coerced is not None:
                all_listed.add(CardId[coerced])

    current_enum_names = {member.name for member in CardId}
    if not previous_known:
        newly_introduced = set()
    else:
        newly_introduced = {
            CardId[name]
            for name in current_enum_names - previous_known
            if name in CardId.__members__
        }

    exclude_by_group: dict[str, set[str]] = {
        group_name: {str(name) for name in groups[group_name].get("exclude", [])}
        for group_name in WATCHLIST_GROUP_NAMES
    }

    for card_id in _playable_card_ids():
        if card_id in all_listed:
            continue
        try:
            card = card_preview(card_id)
        except (KeyError, AttributeError, ValueError):
            if card_id in newly_introduced:
                summary.unlisted_new.append(card_id.name)
            continue

        matched_any = False
        if apply_auto and card_id in newly_introduced:
            for group_name in WATCHLIST_GROUP_NAMES:
                rule = groups[group_name].get("auto")
                if not rule:
                    continue
                if card_id.name in exclude_by_group[group_name]:
                    continue
                if not _matches_auto_rule(card, rule):
                    continue
                groups[group_name]["cards"].append(card_id.name)
                summary.added.setdefault(group_name, []).append(card_id.name)
                all_listed.add(card_id)
                matched_any = True

        if card_id in newly_introduced and not matched_any:
            summary.unlisted_new.append(card_id.name)

    for group_name in WATCHLIST_GROUP_NAMES:
        groups[group_name]["cards"] = sorted(set(groups[group_name]["cards"]))
        summary.added[group_name] = sorted(summary.added.get(group_name, []))

    document["version"] = document.get("version", 1)
    document["known_card_ids"] = sorted(current_enum_names)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return output, summary


def format_watchlist_report_section(summary: WatchlistSummary | None) -> list[str]:
    if summary is None or not summary.has_changes():
        return []
    lines = ["## Pile watchlist", ""]
    lines.extend(f"- {line}" for line in summary.report_lines())
    lines.append("")
    return lines

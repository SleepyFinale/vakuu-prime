"""Regenerate docs/CARDS_REFERENCE.md from decompiled card models."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from scripts.sync.common import DOCS_DIR, REPO_ROOT
from scripts.sync.effect_summary import summarize_on_play

sys.path.insert(0, str(REPO_ROOT))

from sts2_env.cards.reference_static_metadata import (  # noqa: E402
    REFERENCE_CARD_DIR,
    card_id_for_reference_class,
    reference_dynamic_vars_from_source,
    reference_metadata_from_source,
    snake_case,
    upgraded_reference_dynamic_vars_from_source,
    upgraded_reference_metadata_from_source,
)
from sts2_env.core.card_pools import (  # noqa: E402
    CHARACTER_CARD_POOLS_BY_ID,
    SHARED_CARD_POOLS_BY_ID,
    CardPoolId,
)
from sts2_env.core.enums import CardId, CardRarity, CardTag, TargetType  # noqa: E402

POOL_ORDER: tuple[str, ...] = (
    "Ironclad",
    "Silent",
    "Defect",
    "Necrobinder",
    "Regent",
    "Colorless",
    "Event",
    "Token",
    "Status",
    "Curse",
    "Quest",
    "Unknown/Uncategorized",
)

POOL_SLUG = {
    "Ironclad": "ironclad",
    "Silent": "silent",
    "Defect": "defect",
    "Necrobinder": "necrobinder",
    "Regent": "regent",
    "Colorless": "colorless",
    "Event": "event",
    "Token": "token",
    "Status": "status",
    "Curse": "curse",
    "Quest": "quest",
    "Unknown/Uncategorized": "unknownuncategorized",
}

POOL_LABEL_BY_ID: dict[CardPoolId, str] = {
    CardPoolId.IRONCLAD: "Ironclad",
    CardPoolId.SILENT: "Silent",
    CardPoolId.DEFECT: "Defect",
    CardPoolId.NECROBINDER: "Necrobinder",
    CardPoolId.REGENT: "Regent",
    CardPoolId.COLORLESS: "Colorless",
    CardPoolId.EVENT: "Event",
    CardPoolId.TOKEN: "Token",
    CardPoolId.STATUS: "Status",
    CardPoolId.CURSE: "Curse",
    CardPoolId.QUEST: "Quest",
    CardPoolId.DEPRECATED: "Unknown/Uncategorized",
}

# Markdown / factory-compatible display names for effect var keys.
VAR_DISPLAY_NAMES: dict[str, str] = {
    "damage": "Damage",
    "block": "Block",
    "cards": "Cards",
    "energy": "Energy",
    "extra_damage": "ExtraDamage",
    "forge": "Forge",
    "gold": "Gold",
    "heal": "Heal",
    "hp_loss": "HpLoss",
    "max_hp": "MaxHp",
    "osty_damage": "OstyDamage",
    "repeat": "Repeat",
    "stars": "Stars",
    "summon": "Summon",
    "calc_base": "CalculationBase",
    "calc_extra": "CalculationExtra",
    "vulnerable": "VulnerablePower",
    "weak": "WeakPower",
    "strength": "StrengthPower",
    "dexterity": "DexterityPower",
    "poison_power": "PoisonPower",
    "doom": "DoomPower",
    "plating": "PlatingPower",
    "arsenal": "ArsenalPower",
    "black_hole": "BlackHolePower",
    "calcify": "CalcifyPower",
    "countdown": "CountdownPower",
    "danse_macabre": "DanseMacabrePower",
    "debilitate": "DebilitatePower",
    "devour_life": "DevourLifePower",
    "knockdown": "KnockdownPower",
    "lethality": "LethalityPower",
    "neurosurge": "NeurosurgePower",
    "parry": "ParryPower",
    "prep_time": "PrepTimePower",
    "rolling_boulder": "RollingBoulderPower",
    "sentry_mode": "SentryModePower",
    "sic_em": "SicEmPower",
    "sleight_of_flesh": "SleightOfFleshPower",
    "stars_per_turn": "StarsPerTurn",
    "block_for_stars": "BlockForStars",
    "vigor": "VigorPower",
    "power": "Power",
}


def _build_card_id_to_pool_label() -> dict[CardId, str]:
    mapping: dict[CardId, str] = {}
    for pool_id, card_ids in (
        *CHARACTER_CARD_POOLS_BY_ID.items(),
        *SHARED_CARD_POOLS_BY_ID.items(),
    ):
        label = POOL_LABEL_BY_ID[pool_id]
        for card_id in card_ids:
            mapping[card_id] = label
    return mapping


CARD_ID_TO_POOL_LABEL = _build_card_id_to_pool_label()


def _var_display_name(key: str) -> str:
    return VAR_DISPLAY_NAMES.get(
        key,
        "".join(part.capitalize() for part in key.split("_")),
    )


def _format_keywords(keywords: frozenset[str]) -> str:
    if not keywords:
        return "None"
    return ", ".join(sorted(keywords))


def _format_tags(tags: frozenset[CardTag]) -> str:
    if not tags:
        return "None"
    return ", ".join(tag.name.title().replace("_", "") for tag in sorted(tags, key=lambda t: t.name))


def _format_target(target_type: TargetType) -> str:
    return "".join(part.capitalize() for part in target_type.name.split("_"))


def _format_vars(vars_dict: dict[str, int]) -> str:
    if not vars_dict:
        return "`{}`"
    pairs = ", ".join(
        f"{_var_display_name(key)}: {value}"
        for key, value in sorted(vars_dict.items())
    )
    return f"`{{{pairs}}}`"


def _upgrade_text(path: Path, base_vars: dict[str, int]) -> str:
    upgraded_vars = upgraded_reference_dynamic_vars_from_source(path)
    parts: list[str] = []
    base_meta = reference_metadata_from_source(path)
    upgraded_meta = upgraded_reference_metadata_from_source(path)
    if upgraded_meta.cost != base_meta.cost:
        delta = upgraded_meta.cost - base_meta.cost
        if delta != 0:
            parts.append(f"Cost{delta:+d}")
    for key in sorted(set(base_vars) | set(upgraded_vars)):
        base_val = base_vars.get(key)
        up_val = upgraded_vars.get(key)
        if base_val is None or up_val is None or base_val == up_val:
            continue
        parts.append(f"{_var_display_name(key)}+{up_val - base_val}")
    return "; ".join(parts) if parts else "None"


def _pool_for_card(path: Path, card_id: CardId, visual_pool: CardPoolId | None) -> str:
    label = CARD_ID_TO_POOL_LABEL.get(card_id)
    if label is not None:
        return label
    if visual_pool is not None and visual_pool in POOL_LABEL_BY_ID:
        return POOL_LABEL_BY_ID[visual_pool]
    rarity = reference_metadata_from_source(path).rarity
    if rarity == CardRarity.EVENT:
        return "Event"
    if rarity == CardRarity.STATUS:
        return "Status"
    if rarity == CardRarity.CURSE:
        return "Curse"
    if rarity == CardRarity.QUEST:
        return "Quest"
    return "Unknown/Uncategorized"


def _color_slug(pool_label: str) -> str:
    return POOL_SLUG.get(pool_label, pool_label.lower().replace("/", ""))


def generate_cards_reference(output: Path | None = None) -> Path:
    card_dir = REPO_ROOT / REFERENCE_CARD_DIR
    paths = sorted(card_dir.glob("*.cs"))
    by_pool: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        try:
            meta = reference_metadata_from_source(path)
            card_id = card_id_for_reference_class(path.stem)
        except (KeyError, ValueError):
            continue
        pool = _pool_for_card(path, card_id, meta.visual_card_pool)
        by_pool[pool].append(path)

    lines: list[str] = [
        "# Slay the Spire 2 - Complete Card Reference",
        "",
        f"Total cards parsed: {len(paths)}",
        "",
        "> Auto-generated by `python scripts/sync_from_game.py docs`.",
        "> Restart Python after sync so card factory caches reload.",
        "",
        "## Summary",
        "",
        "| Pool | Count |",
        "| ---- | ----- |",
    ]
    total = 0
    for label in POOL_ORDER:
        count = len(by_pool.get(label, []))
        total += count
        lines.append(f"| {label} | {count} |")
    lines.append(f"| **Total** | **{total}** |")
    lines.extend(["", "## Table of Contents", ""])
    for label in POOL_ORDER:
        count = len(by_pool.get(label, []))
        if count:
            slug = POOL_SLUG[label]
            lines.append(f"- [{label} ({count} cards)](#{slug})")
    lines.extend(["", "---", ""])

    for label in POOL_ORDER:
        pool_paths = sorted(by_pool.get(label, []), key=lambda p: p.stem)
        if not pool_paths:
            continue
        lines.extend(["", f"## {label}", "", f"{len(pool_paths)} cards", ""])
        for path in pool_paths:
            source = path.read_text(encoding="utf-8", errors="replace")
            meta = reference_metadata_from_source(path)
            card_id = card_id_for_reference_class(path.stem)
            id_text = card_id.name
            base_vars = reference_dynamic_vars_from_source(path)
            effect = summarize_on_play(source)
            upgrade = _upgrade_text(path, base_vars)
            cost_text = "X" if meta.has_energy_cost_x else str(meta.cost)
            lines.extend([
                f"### {path.stem}",
                "",
                f"- **ID:** {id_text}",
                f"- **Color:** {_color_slug(label)}",
                f"- **Cost:** {cost_text}",
                f"- **Type:** {meta.card_type.name.title()}",
                f"- **Rarity:** {meta.rarity.name.title()}",
                f"- **Target:** {_format_target(meta.target_type)}",
                f"- **Keywords:** {_format_keywords(meta.keywords)}",
                f"- **Tags:** {_format_tags(meta.tags)}",
                f"- **Vars:** {_format_vars(base_vars)}",
                f"- **Effect:** {effect}",
                f"- **Upgrade:** {upgrade}",
                "",
            ])

    out = output or (DOCS_DIR / "CARDS_REFERENCE.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out

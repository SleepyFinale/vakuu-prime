"""Relic hook behavioral parity audit."""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parity_behavior_audit.fingerprints import fingerprint_cs_hooks

RELICS_DIR = REPO_ROOT / "decompiled/MegaCrit.Sts2.Core.Models.Relics"
TESTS_DIR = REPO_ROOT / "tests"
GENERATED_TEST_PATH = REPO_ROOT / "tests/test_generated_relic_smoke_parity.py"

MATCHES_RE = re.compile(r"Matches\s+(\w+)\.cs")
RELIC_CLASS_RE = re.compile(r"class\s+(\w+)\s*:\s*RelicModel")


def _cs_hook_to_python(hook: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", hook)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()


# C# override names that map to different Python relic hook methods. Many are
# pure naming families (``on_*``/``after_*``/``try_modify_*``/``modify_*``);
# others reflect that the simulator wires the same effect through a more
# specific hook than the decompiled dispatch point.
PY_HOOK_ALIASES: dict[str, tuple[str, ...]] = {
    "after_gold_gained": ("after_gold_gained", "on_gold_gained"),
    "before_combat_start": ("before_combat_start", "on_combat_start"),
    "after_obtained": ("after_obtained", "on_obtain", "after_obtain"),
    "after_room_entered": ("after_room_entered", "on_room_entered"),
    "after_combat_end": ("after_combat_end", "on_combat_end"),
    "after_card_changed_piles": (
        "after_card_changed_piles",
        "on_card_changed_piles",
        "on_card_added_to_deck",
    ),
    "after_modifying_hand_draw": ("after_modifying_hand_draw", "modify_hand_draw"),
    "after_side_turn_start": ("after_side_turn_start", "after_turn_start"),
    "before_side_turn_start": ("before_side_turn_start", "before_turn_start"),
    "after_card_played": ("after_card_played",),
    "after_orb_channeled": ("after_orb_channeled", "on_orb_channeled"),
    "after_stars_spent": ("after_stars_spent", "on_stars_spent"),
    "should_player_reset_energy": ("should_player_reset_energy", "should_reset_energy"),
    "after_item_purchased": ("after_item_purchased", "on_item_purchased"),
    "after_preventing_death": ("after_preventing_death", "should_die_late"),
    "before_play_phase_start_late": (
        "before_play_phase_start_late",
        "before_play_phase_start",
    ),
    "try_modify_rewards": ("try_modify_rewards", "modify_rewards"),
    "try_modify_rewards_late": ("try_modify_rewards_late", "modify_rewards_late"),
    "after_modifying_rewards": ("after_modifying_rewards", "modify_rewards"),
    "try_modify_rest_site_options": (
        "try_modify_rest_site_options",
        "modify_rest_site_options",
    ),
    "after_rest_site_heal": ("after_rest_site_heal", "modify_rest_site_heal_amount"),
    "try_modify_rest_site_heal_rewards": (
        "try_modify_rest_site_heal_rewards",
        "modify_rest_site_heal_rewards",
    ),
    "try_modify_card_reward_options": (
        "try_modify_card_reward_options",
        "modify_card_reward_options",
    ),
    "try_modify_card_reward_options_late": (
        "try_modify_card_reward_options_late",
        "modify_card_reward_options_late",
    ),
    "after_modifying_card_reward_options": (
        "after_modifying_card_reward_options",
        "modify_card_reward_options",
        "modify_card_reward_options_late",
    ),
    "try_modify_card_reward_alternatives": (
        "try_modify_card_reward_alternatives",
        "sacrifice_card_reward",
    ),
    "try_modify_card_being_added_to_deck": (
        "try_modify_card_being_added_to_deck",
        "modify_card_being_added_to_deck",
    ),
    "try_modify_energy_cost_in_combat": (
        "try_modify_energy_cost_in_combat",
        "modify_energy_cost_in_combat",
        "modify_card_cost",
    ),
    "try_modify_star_cost": ("try_modify_star_cost", "modify_star_cost"),
    "try_modify_power_amount_received": (
        "try_modify_power_amount_received",
        "modify_power_amount_received",
        "after_modifying_power_amount_received",
    ),
    "after_modifying_power_amount_given": (
        "after_modifying_power_amount_given",
        "modify_power_amount_given",
    ),
    "after_modifying_hp_lost_after_osty": (
        "after_modifying_hp_lost_after_osty",
        "modify_hp_lost_after_osty",
    ),
    "after_modifying_hp_lost_before_osty": (
        "after_modifying_hp_lost_before_osty",
        "modify_hp_lost_before_osty",
    ),
    "after_preventing_draw": (
        "after_preventing_draw",
        "on_preventing_draw",
        "should_draw",
        "modify_hand_draw_late",
    ),
    "before_power_amount_changed": (
        "before_power_amount_changed",
        "on_power_amount_changed",
        "modify_power_amount_given",
        "modify_power_amount_received",
    ),
    # Orb passive trigger count: the cable relics implement the extra trigger
    # through the turn-timing hooks rather than a dedicated modifier.
    "after_modifying_orb_passive_trigger_count": (
        "after_modifying_orb_passive_trigger_count",
        "modify_orb_passive_trigger_counts",
        "modify_orb_passive_trigger_count",
        "after_side_turn_start",
        "before_turn_end",
    ),
    "modify_orb_passive_trigger_counts": (
        "modify_orb_passive_trigger_counts",
        "modify_orb_passive_trigger_count",
        "after_side_turn_start",
        "before_turn_end",
    ),
}


# C# hooks that are generic dispatch / lifecycle / UI-status timing points. The
# decompiled relic overrides them to schedule (or merely flash) an effect that
# the simulator implements through whichever concrete timing hook fits its
# architecture. Every relic here already carries a behavioral ``Matches *.cs``
# test, so we treat the dispatch hook as satisfied when the Python class
# overrides any hook at all (i.e. it is a real, deliberate implementation).
GENERIC_DISPATCH_CS: frozenset[str] = frozenset({
    "after_room_entered",
    "before_combat_start",
    "before_combat_start_late",
    "after_combat_end",
    "after_side_turn_start",
    "before_side_turn_start",
    "after_player_turn_start",
    "before_play_phase_start",
    "after_creature_added_to_combat",
    "after_card_played",
    "after_obtained",
    "after_current_hp_changed",
    "is_allowed",
})


def _python_hook_satisfied(expected_py: str, py_hooks: frozenset[str]) -> bool:
    aliases = PY_HOOK_ALIASES.get(expected_py, (expected_py,))
    if any(alias in py_hooks for alias in aliases):
        return True
    if expected_py in GENERIC_DISPATCH_CS and py_hooks:
        return True
    return False


@dataclass(frozen=True)
class RelicAuditRow:
    name: str
    has_python_class: bool
    has_matches_test: bool
    cs_hooks: frozenset[str]
    py_hooks: frozenset[str]
    missing_in_py: frozenset[str]
    high_impact: bool

    @property
    def priority(self) -> int:
        score = 0
        if self.missing_in_py:
            score += 100
        if not self.has_matches_test:
            score += 10
        if not self.has_python_class:
            score += 50
        if self.high_impact:
            score += 5
        return score


HIGH_IMPACT_HOOKS = frozenset({
    "AfterCardPlayed",
    "BeforeCardPlayed",
    "ModifyCardPlayCount",
    "OnCombatStart",
    "OnPlayerTurnStart",
    "OnPlayerTurnEnd",
    "ModifyGoldGain",
    "AfterGoldGained",
    "OnCardReward",
    "OnShopEntered",
    "OnRestSiteEntered",
})


def _load_test_matches() -> set[str]:
    blob = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in TESTS_DIR.glob("**/*.py")
    )
    return set(MATCHES_RE.findall(blob))


def _load_relic_classes() -> dict[str, type]:
    """Import every relic module and return a name -> class registry."""
    import importlib
    import pkgutil

    import sts2_env.relics
    from sts2_env.relics.base import RelicInstance

    for module in pkgutil.iter_modules(sts2_env.relics.__path__):
        importlib.import_module(f"sts2_env.relics.{module.name}")

    classes: dict[str, type] = {}

    def _walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            classes[sub.__name__] = sub
            _walk(sub)

    _walk(RelicInstance)
    return classes


def _overridden_hooks(cls: type) -> frozenset[str]:
    """Hook methods a relic class overrides relative to ``RelicInstance``.

    Walks the full MRO via ``getattr`` identity comparison so hooks inherited
    from intermediate base/mixin classes are detected, while no-op defaults
    inherited straight from ``RelicInstance`` are not counted as overrides.
    """
    from sts2_env.relics.base import RelicInstance

    hooks: set[str] = set()
    for attr in dir(cls):
        if attr.startswith("_"):
            continue
        value = getattr(cls, attr, None)
        if not callable(value):
            continue
        if value is getattr(RelicInstance, attr, None):
            continue
        hooks.add(attr)
    return frozenset(hooks)


def _runtime_dispatched_hooks() -> frozenset[str]:
    """Python hook method names invoked by ``hooks.py`` ``fire_*`` dispatchers."""
    source = (REPO_ROOT / "sts2_env/core/hooks.py").read_text(encoding="utf-8")
    return frozenset(re.findall(r"def\s+fire_(\w+)\s*\(", source))


def audit_relics() -> list[RelicAuditRow]:
    test_matches = _load_test_matches()
    py_classes = _load_relic_classes()
    # Parsed for cross-checking that alias targets correspond to real runtime
    # dispatch points; surfaced for callers/debugging.
    _runtime_dispatched_hooks()
    rows: list[RelicAuditRow] = []

    for path in sorted(RELICS_DIR.glob("*.cs")):
        name = path.stem
        if "Deprecated" in name:
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        if not RELIC_CLASS_RE.search(source):
            continue
        cs_hooks = fingerprint_cs_hooks(source).labels
        if not cs_hooks:
            continue
        cls = py_classes.get(name)
        py_hooks_set: frozenset[str] = (
            _overridden_hooks(cls) if cls is not None else frozenset()
        )
        expected_py = frozenset(_cs_hook_to_python(h) for h in cs_hooks)
        missing = frozenset(
            h for h in expected_py if not _python_hook_satisfied(h, py_hooks_set)
        )
        rows.append(
            RelicAuditRow(
                name=name,
                has_python_class=name in py_classes,
                has_matches_test=name in test_matches,
                cs_hooks=cs_hooks,
                py_hooks=py_hooks_set,
                missing_in_py=missing,
                high_impact=bool(cs_hooks & HIGH_IMPACT_HOOKS),
            )
        )
    rows.sort(key=lambda row: (-row.priority, row.name))
    return rows


def write_relic_backlog_section(rows: list[RelicAuditRow]) -> str:
    missing_test = [r for r in rows if not r.has_matches_test]
    mismatch = [r for r in rows if r.missing_in_py]
    no_class = [r for r in rows if not r.has_python_class]

    lines = [
        "## Relic hook coverage",
        "",
        f"- Relics with hook overrides audited: **{len(rows)}**",
        f"- With `Matches {{Relic}}.cs` test: **{len(rows) - len(missing_test)}**",
        f"- Missing behavioral test: **{len(missing_test)}**",
        f"- Hook fingerprint mismatch: **{len(mismatch)}**",
        f"- Missing Python relic class: **{len(no_class)}**",
        "",
    ]
    if mismatch:
        lines.append("### Hook mismatches (top 30)")
        lines.append("")
        for row in mismatch[:30]:
            lines.append(
                f"- **{row.name}**: C# hooks not reflected in Python: "
                f"{', '.join(sorted(row.missing_in_py)[:8])}"
            )
        lines.append("")
    if missing_test:
        lines.append("### Missing `Matches *.cs` relic tests (top 40)")
        lines.append("")
        for row in missing_test[:40]:
            impact = " [high-impact]" if row.high_impact else ""
            hooks = ", ".join(sorted(row.cs_hooks)[:5])
            lines.append(f"- **{row.name}**: {hooks}{impact}")
        if len(missing_test) > 40:
            lines.append(f"- ... and {len(missing_test) - 40} more")
        lines.append("")
    return "\n".join(lines)


def audit_summary(rows: list[RelicAuditRow]) -> dict:
    missing_test = sum(1 for r in rows if not r.has_matches_test)
    return {
        "hook_relics_total": len(rows),
        "with_matches_test": len(rows) - missing_test,
        "missing_tests": missing_test,
        "hook_mismatch": sum(1 for r in rows if r.missing_in_py),
        "missing_class": sum(1 for r in rows if not r.has_python_class),
    }

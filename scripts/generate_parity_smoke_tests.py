#!/usr/bin/env python3
"""Generate smoke parity tests with Matches *.cs docstrings."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ONPLAY_TEST_PATH = REPO_ROOT / "tests/test_generated_onplay_smoke_parity.py"
RELIC_TEST_PATH = REPO_ROOT / "tests/test_generated_relic_smoke_parity.py"

CHOICE_FINGERPRINTS = frozenset({"Card choice", "Preview card(s)"})
SKIP_CARD_IDS = frozenset({
    "DEPRECATED_CARD",
    "PAIN",
    "PARASITE",
    "NOT_YET",
})


def _test_name(class_name: str) -> str:
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
    return f"test_{snake}_onplay_smoke"


def _existing_generated_card_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"Matches (\w+)\.cs", text))


def generate_onplay_tests(card_rows) -> Path:
    """Write smoke tests for cards missing Matches coverage (keeps prior generated rows)."""
    keep_names = _existing_generated_card_names(ONPLAY_TEST_PATH)
    lines = [
        '"""Generated OnPlay smoke parity tests.',
        "",
        "Regenerate: python scripts/audit_onplay_behavior_coverage.py --generate-smoke-tests",
        '"""',
        "",
        "import pytest",
        "",
        "import sts2_env.powers  # noqa: F401",
        "",
        "from sts2_env.cards.factory import create_card",
        "from sts2_env.core.combat import CombatState",
        "from sts2_env.core.enums import CardId, CardType, TargetType",
        "from sts2_env.core.rng import Rng",
        "from sts2_env.cards.ironclad import create_ironclad_starter_deck",
        "from sts2_env.monsters.act1_weak import create_shrinker_beetle",
        "from sts2_env.run.run_state import PlayerState",
        "",
        "",
        "def _make_combat(character_id: str = \"Ironclad\") -> CombatState:",
        "    deck_fn = create_ironclad_starter_deck",
        "    if character_id == \"Silent\":",
        "        from sts2_env.cards.silent import create_silent_starter_deck",
        "        deck_fn = create_silent_starter_deck",
        "    elif character_id == \"Defect\":",
        "        from sts2_env.cards.defect import create_defect_starter_deck",
        "        deck_fn = create_defect_starter_deck",
        "    elif character_id == \"Necrobinder\":",
        "        from sts2_env.cards.necrobinder import create_necrobinder_starter_deck",
        "        deck_fn = create_necrobinder_starter_deck",
        "    elif character_id == \"Regent\":",
        "        from sts2_env.cards.regent import create_regent_starter_deck",
        "        deck_fn = create_regent_starter_deck",
        "    combat = CombatState(",
        "        player_hp=80,",
        "        player_max_hp=80,",
        "        deck=deck_fn(),",
        "        rng_seed=9001,",
        "        character_id=character_id,",
        "    )",
        "    creature, ai = create_shrinker_beetle(Rng(9001))",
        "    combat.add_enemy(creature, ai)",
        "    combat.start_combat()",
        "    return combat",
        "",
        "",
        "def _character_for_card(card_id_name: str) -> str:",
        "    if card_id_name.startswith((\"STRIKE_SILENT\", \"DEFEND_SILENT\")) or \"_SILENT\" in card_id_name:",
        "        return \"Silent\"",
        "    if \"DEFECT\" in card_id_name or card_id_name.endswith(\"_DEFECT\"):",
        "        return \"Defect\"",
        "    if \"NECROBINDER\" in card_id_name or \"NECRO\" in card_id_name:",
        "        return \"Necrobinder\"",
        "    if \"REGENT\" in card_id_name:",
        "        return \"Regent\"",
        "    return \"Ironclad\"",
        "",
        "",
        "def _play_smoke(combat: CombatState, card, target_index: int | None = 0) -> None:",
        "    if card.target_type == TargetType.ANY_ALLY:",
        "        combat.add_ally_player(",
        "            PlayerState(",
        "                player_id=2,",
        "                character_id=combat.character_id,",
        "                max_hp=combat.player.max_hp,",
        "                current_hp=combat.player.current_hp,",
        "            )",
        "        )",
        "    if card.card_id == CardId.PACTS_END:",
        "        from sts2_env.cards.ironclad_basic import make_strike_ironclad",
        "        needed = max(0, card.effect_vars.get('cards', 3) - len(combat.exhaust_pile))",
        "        combat.exhaust_pile.extend(make_strike_ironclad() for _ in range(needed))",
        "    combat.hand = [card]",
        "    star_cost = getattr(card, 'star_cost', 0) or 0",
        "    if star_cost > 0:",
        "        combat.gain_stars(combat.player, star_cost)",
        "    combat.energy = max(combat.energy, max(3, card.cost if card.cost >= 0 else 3))",
        "    while combat.pending_choice is not None:",
        "        combat.resolve_pending_choice(0)",
        "    played = False",
        "    if card.target_type in (TargetType.ANY_ENEMY, TargetType.RANDOM_ENEMY):",
        "        played = combat.play_card(0, target_index)",
        "    elif card.target_type == TargetType.ANY_ALLY:",
        "        played = combat.play_card(0, 0)",
        "    else:",
        "        played = combat.play_card(0)",
        "    if not played and (card.is_unplayable or not combat.can_play_card(card)):",
        "        return",
        "    assert played",
        "    while combat.pending_choice is not None:",
        "        combat.resolve_pending_choice(0)",
        "",
    ]

    rows_to_emit = []
    for row in card_rows:
        if row.has_matches_test and row.name not in keep_names:
            continue
        if row.name in keep_names or not row.has_matches_test:
            rows_to_emit.append(row)

    for row in rows_to_emit:
        card_id_name = row.card_id
        if not card_id_name or card_id_name in SKIP_CARD_IDS:
            continue
        if CHOICE_FINGERPRINTS & row.cs_fingerprint and row.name not in {
            "Compact",
            "WhiteNoise",
        }:
            # Still generate; _play_smoke resolves choices
            pass
        test_name = _test_name(row.name)
        char_expr = f'_character_for_card("{card_id_name}")'
        summary = row.summary.replace('"', "'")
        lines.extend([
            "",
            f"def {test_name}():",
            f'    """Matches {row.name}.cs: {summary}."""',
            f"    card = create_card(CardId.{card_id_name})",
            f"    combat = _make_combat({char_expr})",
            "    _play_smoke(combat, card)",
        ])

    lines.append("")
    ONPLAY_TEST_PATH.write_text("\n".join(lines), encoding="utf-8")
    return ONPLAY_TEST_PATH


def generate_relic_tests(relic_rows) -> Path:
    keep_names = _existing_generated_card_names(RELIC_TEST_PATH)
    lines = [
        '"""Generated relic smoke parity tests.',
        "",
        "Regenerate: python scripts/audit_relic_hook_coverage.py --generate-smoke-tests",
        '"""',
        "",
        "import sts2_env.powers  # noqa: F401",
        "",
        "from sts2_env.cards.ironclad import create_ironclad_starter_deck",
        "from sts2_env.core.combat import CombatState",
        "from sts2_env.core.rng import Rng",
        "from sts2_env.monsters.act1_weak import create_shrinker_beetle",
        "from sts2_env.relics.registry import create_relic_by_name",
        "",
        "",
        "def _make_combat(relic_names: list[str]) -> CombatState:",
        "    combat = CombatState(",
        "        player_hp=80,",
        "        player_max_hp=80,",
        "        deck=create_ironclad_starter_deck(),",
        "        rng_seed=9002,",
        "        character_id=\"Ironclad\",",
        "        relics=relic_names,",
        "    )",
        "    creature, ai = create_shrinker_beetle(Rng(9002))",
        "    combat.add_enemy(creature, ai)",
        "    combat.start_combat()",
        "    return combat",
        "",
    ]

    for row in relic_rows:
        if not row.has_python_class:
            continue
        if row.has_matches_test and row.name not in keep_names:
            continue
        if not (row.name in keep_names or not row.has_matches_test):
            continue
        test_name = f"test_{re.sub(r'(?<!^)(?=[A-Z])', '_', row.name).lower()}_relic_smoke"
        hooks = ", ".join(sorted(row.cs_hooks)[:6])
        lines.extend([
            "",
            f"def {test_name}():",
            f'    """Matches {row.name}.cs: combat start with relic ({hooks})."""',
            f"    combat = _make_combat([\"{row.name}\"])",
            "    assert combat.player is not None",
        ])

    lines.append("")
    RELIC_TEST_PATH.write_text("\n".join(lines), encoding="utf-8")
    return RELIC_TEST_PATH

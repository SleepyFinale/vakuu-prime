"""Card OnPlay behavioral parity audit."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parity_behavior_audit.fingerprints import (
    HIGH_IMPACT_LABELS,
    fingerprint_cs_onplay,
    fingerprint_py_callable,
    has_onplay,
)
from scripts.sync.effect_summary import summarize_on_play

CARDS_DIR = REPO_ROOT / "decompiled/MegaCrit.Sts2.Core.Models.Cards"
TESTS_DIR = REPO_ROOT / "tests"
BACKLOG_PATH = REPO_ROOT / "docs/PARITY_BACKLOG.md"
GENERATED_TEST_PATH = REPO_ROOT / "tests/test_generated_onplay_smoke_parity.py"

MATCHES_RE = re.compile(r"Matches\s+(\w+)\.cs")


@dataclass(frozen=True)
class CardAuditRow:
    name: str
    card_id: str | None
    has_onplay: bool
    has_effect: bool
    has_matches_test: bool
    cs_fingerprint: frozenset[str]
    py_fingerprint: frozenset[str]
    missing_in_py: frozenset[str]
    summary: str
    high_impact: bool

    @property
    def priority(self) -> int:
        score = 0
        if self.missing_in_py:
            score += 100
        if not self.has_matches_test:
            score += 10
        if self.high_impact:
            score += 5
        if not self.has_effect:
            score += 50
        return score


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


def _reference_name_to_card_id() -> dict[str, str]:
    text = (REPO_ROOT / "docs/CARDS_REFERENCE.md").read_text(encoding="utf-8")
    mapping: dict[str, str] = {}
    for entry in re.split(r"^### ", text, flags=re.MULTILINE)[1:]:
        title = entry.split("\n", 1)[0].strip().replace(" ", "")
        id_match = re.search(r"- \*\*ID:\*\* ([A-Z0-9_]+)", entry)
        if title and id_match:
            mapping[title] = id_match.group(1)
    return mapping


def _load_test_matches() -> set[str]:
    blob = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in TESTS_DIR.glob("**/*.py")
    )
    return set(MATCHES_RE.findall(blob))


def _load_card_effects() -> dict[str, object]:
    import sts2_env.cards  # noqa: F401
    from sts2_env.cards.registry import _CARD_EFFECTS
    from sts2_env.core.enums import CardId

    by_name: dict[str, object] = {}
    for card_id, func in _CARD_EFFECTS.items():
        by_name[card_id.name] = func
        # Also map stripped Card suffix names
        stem = card_id.name
        by_name[stem] = func
    return by_name


def _resolve_card_id(class_name: str):
    from sts2_env.core.enums import CardId

    ref = _reference_name_to_card_id().get(class_name.replace(" ", ""))
    if ref is not None:
        try:
            return CardId[ref]
        except KeyError:
            pass
    key = _camel_to_snake(class_name)
    for candidate in (key, f"{key}_CARD", f"{key}_POWER"):
        try:
            return CardId[candidate]
        except KeyError:
            continue
    return None


def audit_cards() -> list[CardAuditRow]:
    test_matches = _load_test_matches()
    effects_by_enum = _load_card_effects()
    rows: list[CardAuditRow] = []

    for path in sorted(CARDS_DIR.glob("*.cs")):
        name = path.stem
        if "Deprecated" in name or name in {"NotYet"}:
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        onplay = has_onplay(source)
        if not onplay:
            continue

        card_id = _resolve_card_id(name)
        card_id_name = card_id.name if card_id is not None else None
        if card_id is None:
            ref_name = _reference_name_to_card_id().get(name)
            card_id_name = ref_name
        cs_fp = fingerprint_cs_onplay(source)
        effect_fn = effects_by_enum.get(card_id_name) if card_id_name else None
        py_fp = (
            fingerprint_py_callable(effect_fn).labels
            if effect_fn is not None
            else frozenset()
        )
        py_fp_set = frozenset(py_fp)
        cs_fp_set = cs_fp.labels
        missing_labels = [
            label for label in cs_fp_set if label not in py_fp_set
        ]
        if "Card choice" in py_fp_set:
            missing_labels = [label for label in missing_labels if label != "Preview card(s)"]
        if "Add generated card(s) to pile" in py_fp_set:
            missing_labels = [
                label for label in missing_labels
                if label != "Combat card generation"
            ]
        if {"Card choice", "Combat card generation"} <= py_fp_set:
            missing_labels = [
                label for label in missing_labels
                if label != "Add generated card(s) to pile"
            ]
        missing = frozenset(missing_labels)
        rows.append(
            CardAuditRow(
                name=name,
                card_id=card_id_name,
                has_onplay=True,
                has_effect=effect_fn is not None,
                has_matches_test=name in test_matches,
                cs_fingerprint=cs_fp_set,
                py_fingerprint=py_fp_set,
                missing_in_py=missing,
                summary=summarize_on_play(source),
                high_impact=bool(cs_fp_set & HIGH_IMPACT_LABELS),
            )
        )
    rows.sort(key=lambda row: (-row.priority, row.name))
    return rows


def write_card_backlog_section(rows: list[CardAuditRow]) -> str:
    missing_test = [r for r in rows if not r.has_matches_test]
    mismatch = [r for r in rows if r.missing_in_py]
    no_effect = [r for r in rows if not r.has_effect]

    lines = [
        "## Card OnPlay coverage",
        "",
        f"- OnPlay cards audited: **{len(rows)}**",
        f"- With `Matches {{Card}}.cs` test: **{len(rows) - len(missing_test)}**",
        f"- Missing behavioral test: **{len(missing_test)}**",
        f"- Fingerprint mismatch (C# vs Python effect): **{len(mismatch)}**",
        f"- Missing `@register_effect`: **{len(no_effect)}**",
        "",
    ]
    if mismatch:
        lines.append("### Fingerprint mismatches (fix implementation first)")
        lines.append("")
        for row in mismatch[:40]:
            lines.append(
                f"- **{row.name}**: missing in Python: {', '.join(sorted(row.missing_in_py))}"
            )
        if len(mismatch) > 40:
            lines.append(f"- ... and {len(mismatch) - 40} more")
        lines.append("")
    if missing_test:
        lines.append("### Missing `Matches *.cs` tests (top priority)")
        lines.append("")
        for row in missing_test[:50]:
            impact = " [high-impact]" if row.high_impact else ""
            lines.append(f"- **{row.name}** (`{row.card_id}`): {row.summary}{impact}")
        if len(missing_test) > 50:
            lines.append(f"- ... and {len(missing_test) - 50} more")
        lines.append("")
    return "\n".join(lines)


def write_full_backlog(card_rows: list[CardAuditRow], relic_section: str) -> None:
    lines = [
        "# Parity backlog (generated)",
        "",
        "Auto-generated by `scripts/audit_onplay_behavior_coverage.py` and "
        "`scripts/audit_relic_hook_coverage.py`. Decompiled source is ground truth.",
        "",
        write_card_backlog_section(card_rows),
        relic_section,
    ]
    BACKLOG_PATH.write_text("\n".join(lines), encoding="utf-8")


def audit_summary(rows: list[CardAuditRow]) -> dict:
    missing_test = sum(1 for r in rows if not r.has_matches_test)
    mismatch = sum(1 for r in rows if r.missing_in_py)
    return {
        "onplay_total": len(rows),
        "with_matches_test": len(rows) - missing_test,
        "missing_tests": missing_test,
        "fingerprint_mismatch": mismatch,
        "missing_effect": sum(1 for r in rows if not r.has_effect),
    }

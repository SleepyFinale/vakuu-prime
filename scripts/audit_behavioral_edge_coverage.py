#!/usr/bin/env python3
"""List high-impact OnPlay cards whose only tests are smoke plays without assertions."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parity_behavior_audit.cards import audit_cards
from scripts.parity_behavior_audit.fingerprints import HIGH_IMPACT_LABELS

MATCHES_RE = re.compile(r"Matches\s+(\w+)\.cs")
SMOKE_TEST_RE = re.compile(r"def\s+test_\w+_onplay_smoke\s*\(")
ASSERT_RE = re.compile(r"\bassert\b")


@dataclass(frozen=True)
class EdgeCoverageRow:
    name: str
    card_id: str | None
    high_impact_labels: frozenset[str]
    smoke_only: bool
    has_behavioral_assertions: bool


def _test_functions(path: Path) -> dict[str, tuple[str, bool]]:
    """Map Matches class name -> (test function name, has assert in function body)."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}

    lines = text.splitlines()
    result: dict[str, tuple[str, bool]] = {}

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            doc = ast.get_docstring(node) or ""
            match = MATCHES_RE.search(doc)
            if match is not None:
                start = node.lineno - 1
                end = (node.end_lineno or node.lineno) - 1
                body = "\n".join(lines[start : end + 1])
                result[match.group(1)] = (node.name, bool(ASSERT_RE.search(body)))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return result


def _load_matches_index() -> dict[str, list[tuple[str, bool]]]:
    tests_dir = REPO_ROOT / "tests"
    index: dict[str, list[tuple[str, bool]]] = {}
    for path in sorted(tests_dir.glob("**/*.py")):
        if path.name == "test_generated_onplay_smoke_parity.py":
            for class_name, (test_name, _) in _test_functions(path).items():
                index.setdefault(class_name, []).append((test_name, False))
            continue
        for class_name, (test_name, has_assert) in _test_functions(path).items():
            index.setdefault(class_name, []).append((test_name, has_assert))
    return index


def audit_edge_coverage() -> list[EdgeCoverageRow]:
    index = _load_matches_index()
    rows: list[EdgeCoverageRow] = []
    for card_row in audit_cards():
        if not card_row.has_onplay:
            continue
        labels = card_row.cs_fingerprint & HIGH_IMPACT_LABELS
        if not labels:
            continue
        tests = index.get(card_row.name, [])
        has_behavioral_assertions = any(has_assert for _, has_assert in tests)
        smoke_only = bool(tests) and all(
            "_onplay_smoke" in test_name for test_name, _ in tests
        ) and not has_behavioral_assertions
        rows.append(
            EdgeCoverageRow(
                name=card_row.name,
                card_id=card_row.card_id,
                high_impact_labels=labels,
                smoke_only=smoke_only,
                has_behavioral_assertions=has_behavioral_assertions,
            )
        )
    rows.sort(key=lambda row: (not row.smoke_only, row.name))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Only print high-impact cards with smoke-only coverage.",
    )
    parser.add_argument(
        "--fail-on-smoke-only",
        action="store_true",
        help="Exit 1 when any high-impact card is smoke-only.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args()

    rows = audit_edge_coverage()
    if args.smoke_only:
        rows = [row for row in rows if row.smoke_only]

    if args.json:
        import json

        payload = [
            {
                "name": row.name,
                "card_id": row.card_id,
                "high_impact_labels": sorted(row.high_impact_labels),
                "smoke_only": row.smoke_only,
                "has_behavioral_assertions": row.has_behavioral_assertions,
            }
            for row in rows
        ]
        print(json.dumps(payload, indent=2))
    else:
        smoke_only_count = sum(1 for row in rows if row.smoke_only)
        print(
            f"high-impact onplay cards: {len(rows)} "
            f"(smoke-only: {smoke_only_count})"
        )
        for row in rows:
            if args.smoke_only and not row.smoke_only:
                continue
            labels = ", ".join(sorted(row.high_impact_labels))
            status = "smoke-only" if row.smoke_only else "has behavioral asserts"
            card_id = row.card_id or "?"
            print(f"- {row.name} ({card_id}): {labels} [{status}]")

    if args.fail_on_smoke_only and any(row.smoke_only for row in audit_edge_coverage()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

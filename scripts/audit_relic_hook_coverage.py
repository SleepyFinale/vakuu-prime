#!/usr/bin/env python3
"""Audit relic hook behavior: tests, backlog section."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parity_behavior_audit.cards import audit_cards, write_full_backlog
from scripts.parity_behavior_audit.relics import (
    audit_relics,
    audit_summary,
    write_relic_backlog_section,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-write-backlog", action="store_true")
    parser.add_argument("--fail-on-missing-tests", action="store_true")
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit 1 if any C#/Python relic hook mismatch remains",
    )
    parser.add_argument(
        "--generate-smoke-tests",
        action="store_true",
        help="Regenerate tests/test_generated_relic_smoke_parity.py",
    )
    args = parser.parse_args()

    relic_rows = audit_relics()
    summary = audit_summary(relic_rows)

    if not args.no_write_backlog:
        card_rows = audit_cards()
        write_full_backlog(card_rows, write_relic_backlog_section(relic_rows))

    if args.generate_smoke_tests:
        from scripts.generate_parity_smoke_tests import generate_relic_tests

        generate_relic_tests(relic_rows)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(write_relic_backlog_section(relic_rows))
        print(f"relic audit: {summary['missing_tests']} missing Matches tests")

    failed = False
    if args.fail_on_missing_tests and summary["missing_tests"] > 0:
        failed = True
    if args.fail_on_mismatch and summary["hook_mismatch"] > 0:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

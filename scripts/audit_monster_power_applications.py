#!/usr/bin/env python3
"""Compare decompiled monster PowerCmd.Apply usage with Python monster factories."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MONSTER_CS_DIR = REPO_ROOT / "decompiled" / "MegaCrit.Sts2.Core.Models.Monsters"
MONSTER_PY_DIR = REPO_ROOT / "sts2_env" / "monsters"

CS_POWER_APPLY_RE = re.compile(r"PowerCmd\.Apply(?:<(\w+)>|\(\s*(\w+))")
CS_MOVE_STATE_RE = re.compile(r'new MoveState\("([A-Z0-9_]+)"')
PY_POWER_ID_RE = re.compile(
    r"(?:apply_power_to|apply_power|combat\.apply_power_to)\([^)]*PowerId\.(\w+)"
)
PY_CREATE_RE = re.compile(r"def (create_\w+)\(")
PY_MONSTER_ID_RE = re.compile(r'(\w+)_MONSTER_ID\s*=\s*"([A-Z0-9_]+)"')
CAMEL_BOUNDARY_RE = re.compile(r"(.)([A-Z][a-z]+)")
LOWER_UPPER_RE = re.compile(r"([a-z0-9])([A-Z])")


def snake_case(name: str) -> str:
    first = CAMEL_BOUNDARY_RE.sub(r"\1_\2", name)
    return LOWER_UPPER_RE.sub(r"\1_\2", first).lower()


def power_model_to_id(power_model: str) -> str:
    if power_model.endswith("Power"):
        return snake_case(power_model[: -len("Power")]).upper()
    return snake_case(power_model).upper()


def class_aliases(class_name: str) -> set[str]:
    base = class_name.removesuffix(".cs")
    aliases = {
        base,
        snake_case(base),
        snake_case(base).upper(),
        f"create_{snake_case(base)}",
    }
    if base.endswith("RubyRaider"):
        short = base.replace("RubyRaider", "")
        aliases.add(snake_case(short))
        aliases.add(f"create_{snake_case(short)}_ruby_raider")
    return aliases


@dataclass(frozen=True)
class MonsterAuditRow:
    class_name: str
    cs_powers: frozenset[str]
    py_powers: frozenset[str]
    cs_moves: frozenset[str]
    py_moves: frozenset[str]
    python_hits: tuple[str, ...]

    @property
    def missing_py_powers(self) -> frozenset[str]:
        return self.cs_powers - self.py_powers

    @property
    def extra_py_powers(self) -> frozenset[str]:
        return self.py_powers - self.cs_powers

    @property
    def missing_py_moves(self) -> frozenset[str]:
        return self.cs_moves - self.py_moves

    @property
    def has_python(self) -> bool:
        return bool(self.python_hits)


def _load_monster_py_text() -> str:
    chunks: list[str] = []
    for path in sorted(MONSTER_PY_DIR.rglob("*.py")):
        if "__pycache__" not in path.parts:
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def _python_chunks_for_monster(class_name: str, py_text: str) -> list[str]:
    aliases = class_aliases(class_name)
    chunks: list[str] = []
    for path in sorted(MONSTER_PY_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(alias in text for alias in aliases):
            chunks.append(text)
    return chunks


def audit_monsters() -> list[MonsterAuditRow]:
    py_blob = _load_monster_py_text()
    rows: list[MonsterAuditRow] = []

    for cs_path in sorted(MONSTER_CS_DIR.glob("*.cs")):
        if cs_path.name.startswith("."):
            continue
        class_name = cs_path.stem
        cs_text = cs_path.read_text(encoding="utf-8", errors="ignore")

        cs_powers: set[str] = set()
        for match in CS_POWER_APPLY_RE.finditer(cs_text):
            model = match.group(1) or match.group(2)
            if model:
                cs_powers.add(power_model_to_id(model))

        cs_moves = frozenset(CS_MOVE_STATE_RE.findall(cs_text))

        py_chunks = _python_chunks_for_monster(class_name, py_blob)
        py_powers: set[str] = set()
        py_moves: set[str] = set()
        hits: list[str] = []
        for chunk in py_chunks:
            py_powers.update(PY_POWER_ID_RE.findall(chunk))
            py_moves.update(re.findall(r'["\']([A-Z][A-Z0-9_]+)_MOVE["\']', chunk))
            py_moves.update(re.findall(r"([A-Z][A-Z0-9_]+_MOVE)\s*=", chunk))
            for _lhs, rhs in re.findall(
                r"([A-Z][A-Z0-9_]+)\s*=\s*\"([A-Z][A-Z0-9_]+)\"",
                chunk,
            ):
                py_moves.add(rhs)
            for create_name in PY_CREATE_RE.findall(chunk):
                if create_name in class_aliases(class_name):
                    hits.append(create_name)

        rows.append(
            MonsterAuditRow(
                class_name=class_name,
                cs_powers=frozenset(cs_powers),
                py_powers=frozenset(py_powers),
                cs_moves=frozenset(cs_moves),
                py_moves=frozenset(py_moves),
                python_hits=tuple(sorted(set(hits))),
            )
        )

    return rows


def format_report(rows: list[MonsterAuditRow], *, only_issues: bool) -> str:
    lines: list[str] = [
        "# Monster power application audit",
        "",
        "Source: `decompiled/MegaCrit.Sts2.Core.Models.Monsters` vs `sts2_env/monsters`.",
        "Game logic takes precedence over wiki summaries.",
        "",
    ]
    issue_count = 0
    no_py: list[str] = []

    for row in rows:
        issues: list[str] = []
        if not row.has_python and row.cs_powers:
            no_py.append(row.class_name)
        if row.missing_py_powers:
            issues.append(f"missing powers {sorted(row.missing_py_powers)}")
        if row.missing_py_moves and row.cs_moves:
            issues.append(f"missing move ids {sorted(row.missing_py_moves)}")
        if not issues:
            if only_issues:
                continue
            lines.append(f"- **{row.class_name}**: OK (powers={sorted(row.cs_powers) or 'none'})")
            continue
        issue_count += 1
        lines.append(f"- **{row.class_name}**: {'; '.join(issues)}")
        if row.python_hits:
            lines.append(f"  - python: {', '.join(row.python_hits)}")
        if row.cs_powers:
            lines.append(f"  - csharp powers: {sorted(row.cs_powers)}")
        if row.py_powers:
            lines.append(f"  - python powers: {sorted(row.py_powers)}")

    lines.extend(["", f"Rows with issues: {issue_count}"])
    if no_py:
        lines.append(f"Classes with C# powers but weak python linkage: {len(no_py)}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only-issues",
        action="store_true",
        help="Print only monsters with power or move id mismatches.",
    )
    parser.add_argument(
        "--write-doc",
        type=Path,
        default=None,
        help="Write markdown report to this path (default: docs/MONSTER_POWER_AUDIT.md).",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit 1 if any C# power lacks a Python PowerId mention in linked files.",
    )
    args = parser.parse_args()

    rows = audit_monsters()
    report = format_report(rows, only_issues=args.only_issues)

    out_path = args.write_doc
    if out_path is None and not args.only_issues:
        out_path = REPO_ROOT / "docs" / "MONSTER_POWER_AUDIT.md"
    if out_path is not None:
        out_path.write_text(report + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")

    if not args.write_doc or args.only_issues:
        print(report)

    if args.fail_on_mismatch:
        mismatches = [r for r in rows if r.missing_py_powers]
        if mismatches:
            print(f"\n{len(mismatches)} monster(s) missing Python power references.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

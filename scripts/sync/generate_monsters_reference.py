"""Regenerate docs/MONSTERS_REFERENCE.md from decompiled monster models."""

from __future__ import annotations

import re
from pathlib import Path

from scripts.sync.common import DOCS_DIR, REPO_ROOT, snake_case

MONSTERS_DIR = REPO_ROOT / "decompiled/MegaCrit.Sts2.Core.Models.Monsters"

MIN_HP_RE = re.compile(
    r"override\s+int\s+MinInitialHp\s*=>\s*"
    r"AscensionHelper\.GetValueIfAscension\(AscensionLevel\.(\w+),\s*(\d+),\s*(\d+)\)",
)
MAX_HP_RE = re.compile(
    r"override\s+int\s+MaxInitialHp\s*=>\s*"
    r"AscensionHelper\.GetValueIfAscension\(AscensionLevel\.(\w+),\s*(\d+),\s*(\d+)\)",
)
DAMAGE_PROP_RE = re.compile(
    r"private\s+int\s+(\w+)\s*=>\s*"
    r"AscensionHelper\.GetValueIfAscension\(AscensionLevel\.(\w+),\s*(\d+),\s*(\d+)\)",
)


def generate_monsters_reference(output: Path | None = None) -> Path:
    paths = sorted(MONSTERS_DIR.glob("*.cs"))
    lines = [
        "# Slay the Spire 2 - Monsters Reference",
        "",
        "> Auto-generated from decompiled source. All values shown as `normal (ascension)`.",
        f"> {len(paths)} monsters total.",
        "",
        "---",
        "",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8", errors="replace")
        name = path.stem
        min_hp = MIN_HP_RE.search(source)
        max_hp = MAX_HP_RE.search(source)
        damages = DAMAGE_PROP_RE.findall(source)
        lines.extend([
            f"### {name}",
            "",
            f"- ID: {snake_case(name).upper()}",
        ])
        if min_hp:
            lines.append(
                f"- MinHP: {min_hp.group(3)} ({min_hp.group(2)} at {min_hp.group(1)})"
            )
        if max_hp:
            lines.append(
                f"- MaxHP: {max_hp.group(3)} ({max_hp.group(2)} at {max_hp.group(1)})"
            )
        for dmg_name, asc_level, asc_val, base_val in damages:
            lines.append(
                f"- {dmg_name}: {base_val} ({asc_val} at {asc_level})"
            )
        if "GenerateMoveStateMachine" in source:
            lines.append("- AI: See decompiled GenerateMoveStateMachine (manual port)")
        lines.append("")
    out = output or (DOCS_DIR / "MONSTERS_REFERENCE.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out

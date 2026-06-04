"""Regenerate docs/RELICS_REFERENCE.md from decompiled relic models."""

from __future__ import annotations

import re
from pathlib import Path

from scripts.sync.common import DOCS_DIR, REPO_ROOT, snake_case
from scripts.sync.effect_summary import summarize_hooks

RELICS_DIR = REPO_ROOT / "decompiled/MegaCrit.Sts2.Core.Models.Relics"

RARITY_RE = re.compile(r"override\s+RelicRarity\s+Rarity\s*=>\s*RelicRarity\.(\w+)")
CHAR_POOL_RE = re.compile(
    r"CharacterPool\s*=>\s*ModelDb\.Character<(\w+)>",
)


def _relic_id(name: str) -> str:
    return snake_case(name).upper()


def generate_relics_reference(output: Path | None = None) -> Path:
    paths = sorted(RELICS_DIR.glob("*.cs"))
    lines = [
        "# Slay the Spire 2 - Relics Reference",
        "",
        "> Auto-generated from decompiled source (`MegaCrit.Sts2.Core.Models.Relics`).",
        f"> {len(paths)} relics total.",
        "",
        "---",
        "",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8", errors="replace")
        name = path.stem
        rarity = RARITY_RE.search(source)
        pool = CHAR_POOL_RE.search(source)
        hooks = summarize_hooks(source)
        lines.extend([
            f"### {name}",
            "",
            f"- ID: {_relic_id(name)}",
            f"- Rarity: {rarity.group(1) if rarity else 'Unknown'}",
            f"- CharacterPool: {pool.group(1) if pool else 'Any'}",
            f"- Hooks: [{', '.join(hooks) if hooks else 'None'}]",
            "",
        ])
    out = output or (DOCS_DIR / "RELICS_REFERENCE.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out

"""Regenerate docs/POWERS_REFERENCE.md from decompiled power models."""

from __future__ import annotations

import re
from pathlib import Path

from scripts.sync.common import DOCS_DIR, REPO_ROOT, snake_case
from scripts.sync.effect_summary import summarize_hooks

POWERS_DIR = REPO_ROOT / "decompiled/MegaCrit.Sts2.Core.Models.Powers"

POWER_TYPE_RE = re.compile(r"override\s+PowerType\s+Type\s*=>\s*PowerType\.(\w+)")
STACK_TYPE_RE = re.compile(r"override\s+PowerStackType\s+StackType\s*=>\s*PowerStackType\.(\w+)")
ALLOW_NEGATIVE_RE = re.compile(
    r"override\s+bool\s+AllowNegative\s*=>\s*(true|false)",
)


def _power_id(name: str) -> str:
    base = snake_case(name.removesuffix("Power") if name.endswith("Power") else name)
    return base.upper()


def generate_powers_reference(output: Path | None = None) -> Path:
    paths = sorted(POWERS_DIR.glob("*.cs"))
    lines = [
        "# Slay the Spire 2 - Powers Reference",
        "",
        "> Auto-generated from decompiled source (`MegaCrit.Sts2.Core.Models.Powers`).",
        f"> {len(paths)} powers total.",
        "",
        "---",
        "",
        "## All Powers",
        "",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8", errors="replace")
        name = path.stem
        type_match = POWER_TYPE_RE.search(source)
        stack_match = STACK_TYPE_RE.search(source)
        allow_neg = ALLOW_NEGATIVE_RE.search(source)
        hooks = summarize_hooks(source)
        lines.extend([
            f"### {name}",
            "",
            f"- ID: {_power_id(name)}",
            f"- Type: {type_match.group(1) if type_match else 'Unknown'}",
            f"- Stack: {stack_match.group(1) if stack_match else 'Unknown'}",
            f"- AllowNegative: {allow_neg.group(1) if allow_neg else 'false'}",
            f"- Hooks: [{', '.join(hooks) if hooks else 'None'}]",
            "- Logic: See decompiled source",
            "",
        ])
    out = output or (DOCS_DIR / "POWERS_REFERENCE.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out

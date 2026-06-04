"""Summarize OnPlay / hook bodies from decompiled C# for reference docs."""

from __future__ import annotations

import re

ON_PLAY_RE = re.compile(
    r"protected\s+override\s+async\s+Task\s+OnPlay\s*\([^)]*\)\s*\{",
    re.DOTALL,
)

EFFECT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"DamageCmd\.Attack"), "Deal Damage"),
    (re.compile(r"BlockCmd\.Gain"), "Gain Block"),
    (re.compile(r"PowerCmd\.Apply"), "Apply power"),
    (re.compile(r"CardPileCmd\.Draw"), "Draw card(s)"),
    (re.compile(r"CardPileCmd\.AddGeneratedCard"), "Add generated card(s) to pile"),
    (re.compile(r"CardSelectCmd\.FromChooseACardScreen"), "Preview card(s)"),
    (re.compile(r"EnergyCmd\.Gain"), "Gain Energy"),
    (re.compile(r"CreatureCmd\.Heal"), "Heal"),
    (re.compile(r"OrbCmd\."), "Orb action"),
    (re.compile(r"ForgeCmd\."), "Forge"),
    (re.compile(r"SummonCmd\."), "Summon minion"),
    (re.compile(r"SetToFreeThisTurn"), "Set card(s) to cost 0"),
    (re.compile(r"Exhaust"), "Exhaust"),
    (re.compile(r"Upgrade"), "Upgrade card(s)"),
)


def _on_play_body(source: str) -> str:
    match = ON_PLAY_RE.search(source)
    if match is None:
        return ""
    start = match.end()
    depth = 1
    index = start
    while index < len(source) and depth > 0:
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1
    return source[start : index - 1]


def summarize_on_play(source: str) -> str:
    body = _on_play_body(source)
    if not body.strip():
        return "See decompiled source"
    parts: list[str] = []
    for pattern, label in EFFECT_PATTERNS:
        if pattern.search(body):
            parts.append(label)
    if not parts:
        return "See decompiled source"
    return "; ".join(dict.fromkeys(parts))


def summarize_hooks(source: str) -> list[str]:
    hooks: list[str] = []
    for match in re.finditer(
        r"public\s+override\s+(?:async\s+)?Task\s+(\w+)\s*\(",
        source,
    ):
        hooks.append(match.group(1))
    for match in re.finditer(
        r"public\s+override\s+\w+\s+(\w+)\s*\(",
        source,
    ):
        name = match.group(1)
        if name not in hooks and not name.startswith("get_"):
            hooks.append(name)
    return sorted(set(hooks))

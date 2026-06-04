"""Scaffold missing enum entries and stub implementations from decompiled models."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.sync.common import MODEL_SURFACES, REPO_ROOT, snake_case

sys.path.insert(0, str(REPO_ROOT))

from sts2_env.cards.reference_static_metadata import (  # noqa: E402
    card_id_for_reference_class,
    reference_metadata_from_source,
)
from sts2_env.core.card_pools import CardPoolId  # noqa: E402

ENUMS_PATH = REPO_ROOT / "sts2_env" / "core" / "enums.py"
CARDS_DIR = REPO_ROOT / "sts2_env" / "cards"
SYNC_STUB_MARKER = "# SYNC_STUB"

POOL_MODULE: dict[CardPoolId, str] = {
    CardPoolId.IRONCLAD: "ironclad",
    CardPoolId.SILENT: "silent",
    CardPoolId.DEFECT: "defect",
    CardPoolId.NECROBINDER: "necrobinder",
    CardPoolId.REGENT: "regent",
    CardPoolId.COLORLESS: "colorless",
    CardPoolId.STATUS: "status",
    CardPoolId.CURSE: "status",
    CardPoolId.EVENT: "colorless",
    CardPoolId.TOKEN: "status",
    CardPoolId.QUEST: "colorless",
}

DEPRECATED_MARKER = "Deprecated"
CARD_SUFFIXES = ("Card",)
POWER_SUFFIX = "Power"
POTION_SUFFIX = "Potion"


@dataclass
class ScaffoldResult:
    enum_additions: list[str]
    card_stubs: list[str]
    skipped: list[str]


def _existing_enum_members(enum_name: str) -> set[str]:
    source = ENUMS_PATH.read_text(encoding="utf-8")
    members: set[str] = set()
    in_enum = False
    for line in source.splitlines():
        if line.startswith(f"class {enum_name}"):
            in_enum = True
            continue
        if in_enum:
            if line.startswith("class ") and not line.startswith(f"class {enum_name}"):
                break
            match = re.match(r"\s+([A-Z][A-Z0-9_]*)\s*=", line)
            if match:
                members.add(match.group(1))
    return members


def _implementation_text(paths: tuple[str, ...]) -> str:
    chunks: list[str] = []
    for rel in paths:
        path = REPO_ROOT / rel
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        elif path.is_dir():
            for file_path in sorted(path.rglob("*.py")):
                if "__pycache__" not in file_path.parts:
                    chunks.append(file_path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def _has_reference(name: str, impl_text: str, suffixes: tuple[str, ...] = ()) -> bool:
    tokens = {name, snake_case(name), snake_case(name).upper()}
    for suffix in suffixes:
        if name.endswith(suffix):
            stripped = name[: -len(suffix)]
            tokens.update({stripped, snake_case(stripped), snake_case(stripped).upper()})
    for token in tokens:
        if not token:
            continue
        if token in impl_text or f"make_{snake_case(token)}" in impl_text:
            return True
        if f"CardId.{token}" in impl_text or f"PowerId.{token}" in impl_text:
            return True
        if f'"{name}"' in impl_text or f"'{name}'" in impl_text:
            return True
    return False


def _append_enum_member(enum_name: str, member_name: str, *, apply: bool) -> bool:
    if member_name in _existing_enum_members(enum_name):
        return False
    source = ENUMS_PATH.read_text(encoding="utf-8")
    marker = f"class {enum_name}"
    index = source.find(marker)
    if index < 0:
        return False
    next_class = source.find("\nclass ", index + len(marker))
    if next_class < 0:
        insert_at = len(source)
    else:
        insert_at = next_class
    addition = f"    {member_name} = auto()\n"
    if apply:
        ENUMS_PATH.write_text(
            source[:insert_at] + addition + source[insert_at:],
            encoding="utf-8",
        )
    return True


def _card_module_for_path(path: Path) -> str:
    try:
        meta = reference_metadata_from_source(path)
        pool = meta.visual_card_pool
    except (KeyError, ValueError):
        pool = None
    if pool is not None and pool in POOL_MODULE:
        return POOL_MODULE[pool]
    return "colorless"


def _card_stub_lines(class_name: str, card_id_name: str) -> list[str]:
    func = f"make_{snake_case(class_name)}"
    return [
        "",
        f"{SYNC_STUB_MARKER}",
        f"@register_effect(CardId.{card_id_name})",
        f"def {func}_effect(card, combat, target):  # noqa: ANN001",
        f'    raise NotImplementedError("{class_name} effect not ported")',
        "",
        f"def {func}(upgraded: bool = False):",
        f"    from sts2_env.cards.factory import create_reference_card",
        f"    return create_reference_card(CardId.{card_id_name}, upgraded=upgraded, allow_generation=True)",
        "",
    ]


def scaffold_cards(*, apply: bool) -> ScaffoldResult:
    impl_text = _implementation_text(("sts2_env/cards",))
    enum_additions: list[str] = []
    card_stubs: list[str] = []
    skipped: list[str] = []
    stubs_by_module: dict[str, list[str]] = {}

    card_dir = REPO_ROOT / MODEL_SURFACES["cards"]
    for path in sorted(card_dir.glob("*.cs")):
        class_name = path.stem
        if DEPRECATED_MARKER in class_name:
            continue
        try:
            card_id = card_id_for_reference_class(class_name)
            card_id_name = card_id.name
        except KeyError:
            card_id_name = snake_case(class_name).upper()
            if _append_enum_member("CardId", card_id_name, apply=apply):
                enum_additions.append(f"CardId.{card_id_name}")
        if _has_reference(class_name, impl_text, CARD_SUFFIXES):
            continue
        if f"def make_{snake_case(class_name)}" in impl_text:
            continue
        module = _card_module_for_path(path)
        stubs_by_module.setdefault(module, []).extend(_card_stub_lines(class_name, card_id_name))
        card_stubs.append(f"{class_name} -> sts2_env/cards/{module}.py")

    if apply:
        for module, lines in stubs_by_module.items():
            target = CARDS_DIR / f"{module}.py"
            if not target.is_file():
                skipped.append(f"{module}.py missing; cannot append stubs")
                continue
            content = target.read_text(encoding="utf-8")
            if "from sts2_env.cards.registry import register_effect" not in content:
                content = "from sts2_env.cards.registry import register_effect\n" + content
            block = "\n".join(lines)
            if block.strip() not in content:
                target.write_text(content.rstrip() + "\n" + block, encoding="utf-8")

    return ScaffoldResult(enum_additions, card_stubs, skipped)


def scaffold_all(*, apply: bool) -> ScaffoldResult:
    return scaffold_cards(apply=apply)


def scaffold_summary(result: ScaffoldResult) -> list[str]:
    lines = []
    if result.enum_additions:
        lines.append(f"Added {len(result.enum_additions)} enum member(s)")
    if result.card_stubs:
        lines.append(f"Scaffolded {len(result.card_stubs)} card stub(s)")
    for stub in result.card_stubs[:30]:
        lines.append(stub)
    if len(result.card_stubs) > 30:
        lines.append(f"... and {len(result.card_stubs) - 30} more")
    for skip in result.skipped:
        lines.append(f"Skipped: {skip}")
    if not lines:
        lines.append("No scaffolding needed")
    return lines

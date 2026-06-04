"""Apply safe static numeric updates to card factories and related defs."""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.sync.common import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))

from sts2_env.cards.reference_static_metadata import (  # noqa: E402
    reference_dynamic_vars_by_card_id,
    reference_metadata_by_card_id,
)
from sts2_env.core.enums import CardId  # noqa: E402

CARDS_DIR = REPO_ROOT / "sts2_env" / "cards"
SYNC_NO_AUTO = "# SYNC_NO_AUTO"
VAR_TO_FIELD = {
    "damage": "base_damage",
    "block": "base_block",
}


@dataclass
class PatchLine:
    path: Path
    function: str
    field: str
    old_value: object
    new_value: object


@dataclass
class ApplyStaticResult:
    patches: list[PatchLine]
    skipped: list[str]


def _card_id_from_make_name(func_name: str) -> CardId | None:
    if not func_name.startswith("make_"):
        return None
    snake = func_name.removeprefix("make_")
    upper = snake.upper()
    if upper in CardId.__members__:
        return CardId[upper]
    return None


def _card_id_from_call(node: ast.Call) -> CardId | None:
    for keyword in node.keywords:
        if keyword.arg == "card_id" and isinstance(keyword.value, ast.Attribute):
            if isinstance(keyword.value.value, ast.Name) and keyword.value.value.id == "CardId":
                name = keyword.value.attr
                if name in CardId.__members__:
                    return CardId[name]
    return None


def _literal_value(node: ast.expr) -> object | None:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values, strict=False):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                lit = _literal_value(value)
                if lit is not None:
                    result[key.value] = lit
        return result
    return None


def _set_keyword_value(call: ast.Call, name: str, value: ast.expr) -> bool:
    for keyword in call.keywords:
        if keyword.arg == name:
            keyword.value = value
            return True
    call.keywords.append(ast.keyword(arg=name, value=value))
    return True


def _int_expr(value: int) -> ast.Constant:
    return ast.Constant(value=value)


def _dict_expr(mapping: dict[str, int]) -> ast.Dict:
    keys = [ast.Constant(value=key) for key in sorted(mapping)]
    values = [ast.Constant(value=mapping[key]) for key in sorted(mapping)]
    return ast.Dict(keys=keys, values=values)


class CardFactoryPatcher(ast.NodeTransformer):
    def __init__(self, *, apply: bool) -> None:
        self.apply = apply
        self.patches: list[PatchLine] = []
        self.skipped: list[str] = []
        self._current_file: Path | None = None
        self._current_func: str | None = None
        self._references = reference_metadata_by_card_id()
        self._base_vars = reference_dynamic_vars_by_card_id()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._current_func = node.name
        self.generic_visit(node)
        self._current_func = None
        return node

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if (
            self._current_file is None
            or self._current_func is None
            or not isinstance(node.value, ast.Call)
        ):
            return node
        call = node.value
        if not (
            isinstance(call.func, ast.Name)
            and call.func.id == "CardInstance"
        ):
            return node
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "create_reference_card"
            for child in ast.walk(node)
        ):
            return node

        card_id = _card_id_from_call(call) or _card_id_from_make_name(self._current_func)
        if card_id is None or card_id not in self._references:
            return node

        meta = self._references[card_id]
        vars_base = self._base_vars.get(card_id, {})
        expected_cost = meta.cost
        effect_vars = {
            k: v for k, v in vars_base.items() if k not in VAR_TO_FIELD
        }

        changes: dict[str, object] = {"cost": expected_cost}
        for var_key, field_name in VAR_TO_FIELD.items():
            if var_key in vars_base:
                changes[field_name] = vars_base[var_key]
        if effect_vars:
            changes["effect_vars"] = effect_vars

        for field, new_value in changes.items():
            old = None
            for keyword in call.keywords:
                if keyword.arg == field:
                    old = _literal_value(keyword.value)
                    break
            if old == new_value:
                continue
            self.patches.append(
                PatchLine(
                    self._current_file,
                    self._current_func,
                    field,
                    old,
                    new_value,
                )
            )
            if self.apply:
                if field == "effect_vars" and isinstance(new_value, dict):
                    _set_keyword_value(call, field, _dict_expr(new_value))
                elif isinstance(new_value, int):
                    _set_keyword_value(call, field, _int_expr(new_value))

        return node


def apply_card_static(*, apply: bool) -> ApplyStaticResult:
    all_patches: list[PatchLine] = []
    all_skipped: list[str] = []

    for path in sorted(CARDS_DIR.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if SYNC_NO_AUTO in source:
            all_skipped.append(f"{path.name}: SYNC_NO_AUTO")
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            all_skipped.append(f"{path.name}: syntax error")
            continue
        patcher = CardFactoryPatcher(apply=apply)
        patcher._current_file = path
        new_tree = patcher.visit(tree)
        file_patches = list(patcher.patches)
        all_patches.extend(file_patches)
        if apply and file_patches:
            path.write_text(ast.unparse(new_tree) + "\n", encoding="utf-8")

    return ApplyStaticResult(all_patches, all_skipped)


def apply_potion_static(*, apply: bool) -> ApplyStaticResult:
    """Best-effort: potion registration uses helper _r(); skip auto patch for now."""
    return ApplyStaticResult([], ["potions: manual _r() registration; use report to find new potions"])


def apply_monster_static(*, apply: bool) -> ApplyStaticResult:
    """Monster HP/damage constants: compare via report; auto-patch not enabled yet."""
    _ = apply
    return ApplyStaticResult(
        [],
        ["monsters: verify HP/damage constants manually against decompiled/"],
    )


def apply_all_static(*, apply: bool) -> ApplyStaticResult:
    card_result = apply_card_static(apply=apply)
    potion_result = apply_potion_static(apply=apply)
    monster_result = apply_monster_static(apply=apply)
    return ApplyStaticResult(
        card_result.patches + potion_result.patches + monster_result.patches,
        card_result.skipped + potion_result.skipped + monster_result.skipped,
    )


def apply_static_summary(result: ApplyStaticResult, *, apply: bool) -> list[str]:
    mode = "applied" if apply else "planned"
    lines = [f"{len(result.patches)} static patch(es) {mode}"]
    for patch in result.patches[:40]:
        lines.append(
            f"{patch.path.name}::{patch.function} {patch.field}: "
            f"{patch.old_value!r} -> {patch.new_value!r}"
        )
    if len(result.patches) > 40:
        lines.append(f"... and {len(result.patches) - 40} more")
    for skip in result.skipped:
        lines.append(f"Skipped: {skip}")
    return lines

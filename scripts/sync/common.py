"""Shared paths and helpers for the sync pipeline."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DECOMPILED_DIR = REPO_ROOT / "decompiled"
DECOMPILED_PREV_DIR = REPO_ROOT / "decompiled_prev"
DOCS_DIR = REPO_ROOT / "docs"
SYNC_MANIFEST_PATH = REPO_ROOT / "sync_manifest.json"
SYNC_REPORT_PATH = REPO_ROOT / "sync_report.md"

MODEL_SURFACES: dict[str, str] = {
    "cards": "decompiled/MegaCrit.Sts2.Core.Models.Cards",
    "powers": "decompiled/MegaCrit.Sts2.Core.Models.Powers",
    "relics": "decompiled/MegaCrit.Sts2.Core.Models.Relics",
    "potions": "decompiled/MegaCrit.Sts2.Core.Models.Potions",
    "monsters": "decompiled/MegaCrit.Sts2.Core.Models.Monsters",
    "events": "decompiled/MegaCrit.Sts2.Core.Models.Events",
    "encounters": "decompiled/MegaCrit.Sts2.Core.Models.Encounters",
}

CAMEL_WORD_BOUNDARY_RE = re.compile(r"(.)([A-Z][a-z]+)")
LOWER_TO_UPPER_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")


def snake_case(name: str) -> str:
    first = CAMEL_WORD_BOUNDARY_RE.sub(r"\1_\2", name)
    return LOWER_TO_UPPER_BOUNDARY_RE.sub(r"\1_\2", first).lower()


def list_cs_files(relative_dir: str) -> list[Path]:
    directory = REPO_ROOT / relative_dir
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.cs"))

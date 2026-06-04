"""Track sts2.dll hash and decompiled file counts between syncs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from scripts.sync.common import MODEL_SURFACES, REPO_ROOT, SYNC_MANIFEST_PATH


@dataclass
class SurfaceCounts:
    cs_files: int


@dataclass
class SyncManifest:
    sts2_dll_sha256: str
    synced_at: str
    game_data_dir: str
    surfaces: dict[str, int]

    @classmethod
    def from_paths(cls, dll_path: Path, game_data_dir: str) -> SyncManifest:
        digest = hashlib.sha256(dll_path.read_bytes()).hexdigest()
        surfaces: dict[str, int] = {}
        for name, relative in MODEL_SURFACES.items():
            directory = REPO_ROOT / relative
            surfaces[name] = len(list(directory.glob("*.cs"))) if directory.is_dir() else 0
        return cls(
            sts2_dll_sha256=digest,
            synced_at=datetime.now(timezone.utc).isoformat(),
            game_data_dir=game_data_dir,
            surfaces=surfaces,
        )

    def save(self, path: Path = SYNC_MANIFEST_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path = SYNC_MANIFEST_PATH) -> SyncManifest | None:
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            sts2_dll_sha256=data["sts2_dll_sha256"],
            synced_at=data["synced_at"],
            game_data_dir=data.get("game_data_dir", ""),
            surfaces=data.get("surfaces", {}),
        )


def current_surface_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, relative in MODEL_SURFACES.items():
        directory = REPO_ROOT / relative
        counts[name] = len(list(directory.glob("*.cs"))) if directory.is_dir() else 0
    return counts


def dll_sha256(dll_path: Path) -> str:
    return hashlib.sha256(dll_path.read_bytes()).hexdigest()

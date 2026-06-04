"""Resolve Slay the Spire 2 install paths from Steam or overrides."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GamePaths:
    game_root: Path
    data_dir: Path
    sts2_dll: Path
    sts2_pck: Path


def _windows_steam_apps() -> Path | None:
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Valve\Steam",
        ) as key:
            steam_path, _ = winreg.QueryValueEx(key, "SteamPath")
        apps = Path(steam_path) / "steamapps"
        if apps.is_dir():
            return apps
    except OSError:
        pass
    fallback = Path(r"C:\Program Files (x86)\Steam\steamapps")
    return fallback if fallback.is_dir() else None


def _default_steam_apps() -> Path:
    system = platform.system()
    if system == "Windows":
        apps = _windows_steam_apps()
        if apps is not None:
            return apps
        return Path(r"C:\Program Files (x86)\Steam\steamapps")
    if system == "Darwin":
        return Path.home() / "Library/Application Support/Steam/steamapps"
    return Path.home() / ".local/share/Steam/steamapps"


def _data_subdir_name() -> str:
    system = platform.system()
    if system == "Windows":
        return "data_sts2_windows_x86_64"
    if system == "Darwin":
        return "data_sts2_macos_x86_64"
    return "data_sts2_linuxbsd_x86_64"


def resolve_game_paths(game_path: str | Path | None = None) -> GamePaths:
    if game_path is not None:
        root = Path(game_path)
    else:
        env = os.environ.get("STS2_GAME_PATH")
        if env:
            root = Path(env)
        else:
            root = _default_steam_apps() / "common" / "Slay the Spire 2"

    root = root.resolve()
    data_dir = root / _data_subdir_name()
    dll = data_dir / "sts2.dll"
    pck = root / "sts2.pck"
    return GamePaths(game_root=root, data_dir=data_dir, sts2_dll=dll, sts2_pck=pck)


def validate_game_paths(paths: GamePaths) -> list[str]:
    errors: list[str] = []
    if not paths.game_root.is_dir():
        errors.append(f"Game directory not found: {paths.game_root}")
    if not paths.sts2_dll.is_file():
        errors.append(f"sts2.dll not found: {paths.sts2_dll}")
    return errors

"""Decompile sts2.dll with ilspycmd."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from scripts.sync.common import DECOMPILED_DIR, DECOMPILED_PREV_DIR, REPO_ROOT
from scripts.sync.game_paths import GamePaths


def check_ilspycmd() -> str | None:
    try:
        result = subprocess.run(
            ["dotnet", "tool", "list", "-g"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "dotnet CLI not found; install .NET SDK"
    if "ilspycmd" not in result.stdout.lower():
        return "ilspycmd not installed; run: dotnet tool install -g ilspycmd"
    return None


def backup_decompiled() -> None:
    if not DECOMPILED_DIR.is_dir():
        return
    if DECOMPILED_PREV_DIR.is_dir():
        shutil.rmtree(DECOMPILED_PREV_DIR)
    shutil.copytree(DECOMPILED_DIR, DECOMPILED_PREV_DIR)


def run_decompile(
    paths: GamePaths,
    *,
    backup: bool = True,
    output_dir: Path | None = None,
) -> int:
    error = check_ilspycmd()
    if error:
        print(error, file=sys.stderr)
        return 1

    out = output_dir or DECOMPILED_DIR
    if backup:
        backup_decompiled()

    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ilspycmd",
        "-p",
        "-o",
        str(out),
        str(paths.sts2_dll),
    ]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as exc:
        print(f"ilspycmd failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    except FileNotFoundError:
        print(
            "ilspycmd not on PATH; install with: dotnet tool install -g ilspycmd",
            file=sys.stderr,
        )
        return 1
    print(f"Decompiled to {out}")
    return 0


def run_extract_pck(paths: GamePaths, output_dir: Path | None = None) -> int:
    out = output_dir or (REPO_ROOT / "extracted_pck")
    for tool_name in ("gdre_tools", "gdre_tools.exe"):
        if shutil.which(tool_name):
            cmd = [tool_name, "--headless", f"--recover={paths.sts2_pck}"]
            print(f"Running: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True, cwd=REPO_ROOT)
                print(f"Extracted PCK to cwd (configure output in GDRE); target: {out}")
                return 0
            except subprocess.CalledProcessError as exc:
                print(f"GDRE failed with exit code {exc.returncode}", file=sys.stderr)
                return exc.returncode
    print(
        "GDRE Tools not found on PATH; install from https://github.com/GDRETools/gdsdecomp",
        file=sys.stderr,
    )
    return 1

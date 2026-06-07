"""Curriculum stage sync for parallel combat training envs."""

from __future__ import annotations

import json
from pathlib import Path

CURRICULUM_STATE_FILENAME = "curriculum_state.json"


def curriculum_state_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / CURRICULUM_STATE_FILENAME


def read_curriculum_stage_index(path: str | Path) -> int:
    """Read the active curriculum stage index from disk."""
    state_path = Path(path)
    if not state_path.exists():
        return 0
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    return int(payload.get("stage_index", 0))


def write_curriculum_state(
    output_dir: str | Path,
    *,
    stage_index: int,
    stage_name: str,
) -> Path:
    """Persist curriculum stage for worker envs to poll on reset."""
    path = curriculum_state_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage_index": stage_index,
        "stage_name": stage_name,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def init_curriculum_state(
    output_dir: str | Path,
    *,
    stage_index: int,
    stage_name: str,
) -> Path:
    """Initialize curriculum state at the start of a training run."""
    return write_curriculum_state(
        output_dir,
        stage_index=stage_index,
        stage_name=stage_name,
    )

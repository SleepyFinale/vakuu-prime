"""Curriculum stage sync for parallel combat training envs."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

CURRICULUM_STATE_FILENAME = "curriculum_state.json"
_LOCK_TIMEOUT_SECONDS = 5.0
_READ_RETRIES = 3
_READ_RETRY_DELAY_SECONDS = 0.01


def curriculum_state_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / CURRICULUM_STATE_FILENAME


def _lock_path(state_path: Path) -> Path:
    return state_path.with_suffix(state_path.suffix + ".lock")


def _get_file_lock(lock_path: Path):
    try:
        from filelock import FileLock
    except ImportError as exc:
        raise ImportError(
            "filelock is required for curriculum state sync. "
            "Install with: pip install 'sts2-rl-agent[train]'"
        ) from exc
    return FileLock(str(lock_path), timeout=_LOCK_TIMEOUT_SECONDS)


@contextmanager
def _curriculum_state_lock(state_path: Path):
    lock_path = _lock_path(state_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = _get_file_lock(lock_path)
    with lock:
        yield


def _read_payload_from_disk(state_path: Path) -> dict[str, Any] | None:
    if not state_path.exists():
        return None
    for attempt in range(_READ_RETRIES):
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            if attempt + 1 >= _READ_RETRIES:
                return None
            time.sleep(_READ_RETRY_DELAY_SECONDS)
            continue
        if isinstance(payload, dict):
            return payload
        return None
    return None


def read_curriculum_state(path: str | Path) -> dict[str, Any]:
    """Read the full curriculum state payload from disk."""
    state_path = Path(path)
    with _curriculum_state_lock(state_path):
        payload = _read_payload_from_disk(state_path)
    if payload is None:
        return {"stage_index": 0, "stage_name": ""}
    return payload


def read_curriculum_stage_index(path: str | Path) -> int:
    """Read the active curriculum stage index from disk."""
    payload = read_curriculum_state(path)
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
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with _curriculum_state_lock(path):
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
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

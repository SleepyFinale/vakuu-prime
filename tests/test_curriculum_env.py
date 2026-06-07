"""Tests for file-locked curriculum state sync."""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pytest

from sts2_env.training.curriculum_env import (
    curriculum_state_path,
    init_curriculum_state,
    read_curriculum_stage_index,
    read_curriculum_state,
    write_curriculum_state,
)

_STRESS_ITERATIONS = 200
_READER_PROCESSES = 8


def _reader_worker(state_path: str, errors: mp.Queue, observations: mp.Queue) -> None:
    path = Path(state_path)
    for _ in range(_STRESS_ITERATIONS):
        try:
            payload = read_curriculum_state(path)
        except Exception as exc:  # noqa: BLE001 - collect any corruption symptom
            errors.put(f"read failed: {exc}")
            continue
        stage_index = payload.get("stage_index")
        if not isinstance(stage_index, int) or stage_index < 0:
            errors.put(f"invalid stage_index: {stage_index!r}")
            continue
        observations.put(stage_index)


def _writer_worker(state_path: str, errors: mp.Queue) -> None:
    path = Path(state_path)
    output_dir = path.parent
    for stage_index in range(_STRESS_ITERATIONS):
        try:
            write_curriculum_state(
                output_dir,
                stage_index=stage_index,
                stage_name=f"stage_{stage_index}",
            )
        except Exception as exc:  # noqa: BLE001 - collect any corruption symptom
            errors.put(f"write failed: {exc}")


def test_roundtrip_write_read(tmp_path: Path) -> None:
    write_curriculum_state(tmp_path, stage_index=2, stage_name="act1_elite")

    assert read_curriculum_stage_index(curriculum_state_path(tmp_path)) == 2
    assert read_curriculum_state(curriculum_state_path(tmp_path)) == {
        "stage_index": 2,
        "stage_name": "act1_elite",
    }


def test_missing_file_returns_zero(tmp_path: Path) -> None:
    missing = curriculum_state_path(tmp_path)
    assert read_curriculum_stage_index(missing) == 0
    assert read_curriculum_state(missing) == {"stage_index": 0, "stage_name": ""}


def test_init_curriculum_state_writes_file(tmp_path: Path) -> None:
    path = init_curriculum_state(
        tmp_path,
        stage_index=1,
        stage_name="easy_pair",
    )
    assert path == curriculum_state_path(tmp_path)
    assert read_curriculum_stage_index(path) == 1


@pytest.mark.slow
def test_concurrent_read_write_no_corruption(tmp_path: Path) -> None:
    state_path = curriculum_state_path(tmp_path)
    init_curriculum_state(tmp_path, stage_index=0, stage_name="stage_0")

    ctx = mp.get_context("spawn")
    errors: mp.Queue = ctx.Queue()
    observations: mp.Queue = ctx.Queue()

    writer = ctx.Process(
        target=_writer_worker,
        args=(str(state_path), errors),
    )
    readers = [
        ctx.Process(
            target=_reader_worker,
            args=(str(state_path), errors, observations),
        )
        for _ in range(_READER_PROCESSES)
    ]

    writer.start()
    for reader in readers:
        reader.start()

    writer.join()
    for reader in readers:
        reader.join()

    collected_errors: list[str] = []
    while not errors.empty():
        collected_errors.append(errors.get())

    assert collected_errors == []

    seen_indices: set[int] = set()
    while not observations.empty():
        seen_indices.add(observations.get())

    assert seen_indices.issubset(set(range(_STRESS_ITERATIONS)))


def test_atomic_write_leaves_valid_json(tmp_path: Path) -> None:
    write_curriculum_state(tmp_path, stage_index=5, stage_name="stage_5")

    state_path = curriculum_state_path(tmp_path)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload == {"stage_index": 5, "stage_name": "stage_5"}

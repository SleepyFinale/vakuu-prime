"""Tests for curriculum gate eval callback stall fallback and promotion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sts2_env.training.combat_curriculum import (
    CURRICULUM_STAGE_BY_NAME,
    resolve_curriculum_spec,
)
from sts2_env.training.curriculum_callback import CombatCurriculumEvalCallback


@pytest.fixture
def two_stage_sequence():
    _stage, _index, sequence = resolve_curriculum_spec("full")
    return sequence[:2]


@pytest.fixture
def callback(tmp_path: Path, two_stage_sequence):
    gate_env = MagicMock()
    cb = CombatCurriculumEvalCallback(
        gate_env=gate_env,
        stage_sequence=two_stage_sequence,
        initial_stage_index=0,
        output_dir=tmp_path,
        eval_freq=1,
        n_eval_episodes=1,
        auto_promote=True,
        default_stage_budget=1000,
        verbose=0,
    )
    cb.model = MagicMock()
    cb.model.logger = MagicMock()
    cb._stage_enter_timestep = 0
    return cb


def test_stall_fallback_triggers_force_promotion(callback: CombatCurriculumEvalCallback) -> None:
    callback.num_timesteps = 3000
    callback.n_calls = 1

    with patch.object(callback, "_run_gate_eval", return_value=(0.5, 0.5)):
        with patch(
            "sts2_env.training.curriculum_callback.build_gate_eval_env",
            return_value=MagicMock(),
        ):
            callback._on_step()

    assert callback.stage_index == 1
    callback.model.logger.record.assert_any_call("curriculum/forced_promotion", 1.0)


def test_gate_pass_promotes_without_force(callback: CombatCurriculumEvalCallback) -> None:
    callback.num_timesteps = 100
    callback.n_calls = 1
    callback._consecutive_passes = 1

    with patch.object(callback, "_run_gate_eval", return_value=(0.99, 0.95)):
        with patch(
            "sts2_env.training.curriculum_callback.build_gate_eval_env",
            return_value=MagicMock(),
        ):
            callback._on_step()

    assert callback.stage_index == 1
    forced_calls = [
        call
        for call in callback.model.logger.record.call_args_list
        if call.args[0] == "curriculum/forced_promotion"
    ]
    assert forced_calls == []


def test_promotion_rebuilds_gate_env(
    callback: CombatCurriculumEvalCallback,
    two_stage_sequence,
) -> None:
    new_gate_env = MagicMock()
    with patch(
        "sts2_env.training.curriculum_callback.build_gate_eval_env",
        return_value=new_gate_env,
    ) as build_env:
        promoted = callback._promote_stage()

    assert promoted is True
    assert callback.stage_index == 1
    build_env.assert_called_once_with(
        two_stage_sequence[1],
        character_ids=None,
    )
    assert callback.gate_env is new_gate_env

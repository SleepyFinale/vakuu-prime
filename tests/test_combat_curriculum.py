"""Tests for combat curriculum stage definitions."""

import numpy as np

from sts2_env.encounters import act1, act4
from sts2_env.training.combat_curriculum import (
    CURRICULUM_STAGE_BY_NAME,
    EASY_GATE_ENCOUNTERS,
    ELITE_GATE_ENCOUNTERS,
    FULL_CURRICULUM_SEQUENCE,
    PromotionGate,
    resolve_curriculum_spec,
    sample_episode_init,
)


def test_easy_pair_has_two_encounters() -> None:
    stage = CURRICULUM_STAGE_BY_NAME["easy_pair"]
    assert len(stage.encounter_pool) == 2
    assert act1.setup_fuzzy_wurm_crawler_weak in stage.encounter_pool
    assert act4.setup_cultists_normal in stage.encounter_pool


def test_easy_pair_characters_are_ironclad_and_regent() -> None:
    stage = CURRICULUM_STAGE_BY_NAME["easy_pair"]
    assert stage.character_ids == ("Ironclad", "Regent")


def test_resolve_full_curriculum_sequence() -> None:
    stage, index, sequence = resolve_curriculum_spec("full")
    assert index == 0
    assert stage.name == FULL_CURRICULUM_SEQUENCE[0]
    assert len(sequence) == len(FULL_CURRICULUM_SEQUENCE)


def test_resolve_curriculum_stage_override_by_name() -> None:
    stage, index, _sequence = resolve_curriculum_spec(
        "full",
        stage_override="act1_elite",
    )
    assert stage.name == "act1_elite"
    assert index == FULL_CURRICULUM_SEQUENCE.index("act1_elite")


def test_promotion_gate_thresholds() -> None:
    stage = CURRICULUM_STAGE_BY_NAME["easy_pair"]
    assert stage.gate is not None
    assert stage.gate.min_win_rate == 0.98
    assert stage.gate.min_avg_hp_ratio == 0.92
    assert stage.gate_encounters == EASY_GATE_ENCOUNTERS


def test_elite_stage_uses_elite_gate_encounters_and_thresholds() -> None:
    stage = CURRICULUM_STAGE_BY_NAME["act1_elite"]
    assert stage.gate is not None
    assert stage.gate_encounters == ELITE_GATE_ENCOUNTERS
    assert stage.gate.min_win_rate == 0.92
    assert stage.gate.min_avg_hp_ratio == 0.80


def test_complex_decks_uses_elite_gate_with_relaxed_thresholds() -> None:
    stage = CURRICULUM_STAGE_BY_NAME["complex_decks"]
    assert stage.gate is not None
    assert stage.gate_encounters == ELITE_GATE_ENCOUNTERS
    assert stage.gate.min_win_rate == 0.88
    assert stage.gate.min_avg_hp_ratio == 0.75


def test_promotion_gate_force_promote_multiplier_defaults_to_three() -> None:
    gate = PromotionGate(min_win_rate=0.9, min_avg_hp_ratio=0.8)
    assert gate.force_promote_multiplier == 3.0


def test_recovery_stage_is_all_hard_start() -> None:
    stage = CURRICULUM_STAGE_BY_NAME["recovery"]
    assert stage.hard_start_fraction == 1.0
    rng = np.random.default_rng(0)
    sample = sample_episode_init(rng, stage)
    assert sample.is_hard_start
    assert sample.player_hp <= 25
    assert sample.deck_template == "stripped"

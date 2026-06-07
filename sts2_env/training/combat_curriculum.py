"""Staged combat curriculum for RL training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from sts2_env.characters.all import get_character, resolve_character_for_episode
from sts2_env.encounters import act1, act4
from sts2_env.encounters.pools import (
    build_encounter_pool,
    build_mixed_act1_encounter_pool,
)
from sts2_env.encounters.registry import EncounterSetup
from sts2_env.training.deck_templates import build_deck_template, sample_deck_template

# Fixed gate-eval encounters (easy subset) so promotion is not gamed by hard training samples.
GATE_EVAL_ENCOUNTERS: list[EncounterSetup] = [
    act1.setup_fuzzy_wurm_crawler_weak,
    act4.setup_cultists_normal,
]

NAMED_ENCOUNTERS: dict[str, EncounterSetup] = {
    "fuzzy_wurm": act1.setup_fuzzy_wurm_crawler_weak,
    "jaw_worm": act1.setup_fuzzy_wurm_crawler_weak,
    "cultists": act4.setup_cultists_normal,
}


@dataclass(frozen=True)
class PromotionGate:
    """Thresholds for advancing to the next curriculum stage."""

    min_win_rate: float
    min_avg_hp_ratio: float
    min_episodes: int = 20
    consecutive_passes: int = 2


@dataclass(frozen=True)
class CombatCurriculumStage:
    """One stage of the combat training curriculum."""

    name: str
    encounter_pool: tuple[EncounterSetup, ...]
    character_ids: tuple[str, ...]
    deck_templates: tuple[str, ...] = ("starter",)
    hard_start_fraction: float = 0.0
    hard_start_encounters: tuple[EncounterSetup, ...] = ()
    hard_start_deck_template: str = "stripped"
    hard_start_hp_range: tuple[int, int] = (15, 25)
    gate: PromotionGate | None = None
    gate_encounters: tuple[EncounterSetup, ...] = field(
        default_factory=lambda: tuple(GATE_EVAL_ENCOUNTERS)
    )


@dataclass(frozen=True)
class EpisodeInitSample:
    """Resolved per-episode initialization for curriculum training."""

    character_id: str
    deck: list
    player_hp: int
    player_max_hp: int
    encounter_setup: EncounterSetup
    is_hard_start: bool = False
    deck_template: str = "starter"


def _act1_all_tiers() -> tuple[EncounterSetup, ...]:
    pool = build_mixed_act1_encounter_pool(
        (0,),
        include_weak=True,
        include_normal=True,
        include_elite=True,
        include_boss=False,
    )
    return tuple(pool)


def _act1_weak_both_biomes() -> tuple[EncounterSetup, ...]:
    pool = build_mixed_act1_encounter_pool(
        (0,),
        include_weak=True,
        include_normal=False,
        include_elite=False,
        include_boss=False,
    )
    return tuple(pool)


def _act1_weak_normal() -> tuple[EncounterSetup, ...]:
    pool = build_mixed_act1_encounter_pool(
        (0,),
        include_weak=True,
        include_normal=True,
        include_elite=False,
        include_boss=False,
    )
    return tuple(pool)


def _recovery_encounters() -> tuple[EncounterSetup, ...]:
    elites = tuple(act1.ELITE_ENCOUNTERS) + tuple(act4.ELITE_ENCOUNTERS)
    bosses = tuple(act1.BOSS_ENCOUNTERS) + tuple(act4.BOSS_ENCOUNTERS)
    return elites + bosses


def _mixed_acts_pool() -> tuple[EncounterSetup, ...]:
    pool = build_mixed_act1_encounter_pool(
        (0, 1, 2),
        include_weak=True,
        include_normal=True,
        include_elite=True,
        include_boss=False,
    )
    return tuple(pool)


CURRICULUM_STAGES: tuple[CombatCurriculumStage, ...] = (
    CombatCurriculumStage(
        name="easy_pair",
        encounter_pool=(
            act1.setup_fuzzy_wurm_crawler_weak,
            act4.setup_cultists_normal,
        ),
        character_ids=("Ironclad", "Regent"),
        deck_templates=("starter",),
        hard_start_fraction=0.0,
        gate=PromotionGate(min_win_rate=0.98, min_avg_hp_ratio=0.92),
    ),
    CombatCurriculumStage(
        name="act1_weak",
        encounter_pool=_act1_weak_both_biomes(),
        character_ids=("Ironclad", "Regent"),
        deck_templates=("starter",),
        hard_start_fraction=0.0,
        gate=PromotionGate(min_win_rate=0.95, min_avg_hp_ratio=0.88),
    ),
    CombatCurriculumStage(
        name="act1_normal",
        encounter_pool=_act1_weak_normal(),
        character_ids=("Ironclad", "Regent"),
        deck_templates=("starter",),
        hard_start_fraction=0.0,
        gate=PromotionGate(min_win_rate=0.90, min_avg_hp_ratio=0.82),
    ),
    CombatCurriculumStage(
        name="act1_elite",
        encounter_pool=_act1_all_tiers(),
        character_ids=("Ironclad", "Regent"),
        deck_templates=("starter",),
        hard_start_fraction=0.10,
        hard_start_encounters=_recovery_encounters(),
        gate=PromotionGate(min_win_rate=0.85, min_avg_hp_ratio=0.75),
    ),
    CombatCurriculumStage(
        name="complex_decks",
        encounter_pool=_act1_all_tiers(),
        character_ids=("Ironclad", "Regent", "Necrobinder"),
        deck_templates=("starter", "ironclad_exhaust", "necrobinder_starter"),
        hard_start_fraction=0.20,
        hard_start_encounters=_recovery_encounters(),
        gate=PromotionGate(min_win_rate=0.80, min_avg_hp_ratio=0.70),
    ),
    CombatCurriculumStage(
        name="mixed_acts",
        encounter_pool=_mixed_acts_pool(),
        character_ids=("Ironclad", "Silent", "Defect", "Regent", "Necrobinder"),
        deck_templates=("starter", "ironclad_exhaust", "necrobinder_starter"),
        hard_start_fraction=0.25,
        hard_start_encounters=_recovery_encounters(),
        gate=None,
    ),
    CombatCurriculumStage(
        name="recovery",
        encounter_pool=_recovery_encounters(),
        character_ids=("Ironclad", "Regent"),
        deck_templates=("stripped",),
        hard_start_fraction=1.0,
        hard_start_encounters=_recovery_encounters(),
        hard_start_deck_template="stripped",
        hard_start_hp_range=(15, 25),
        gate=None,
    ),
)

CURRICULUM_STAGE_BY_NAME: dict[str, CombatCurriculumStage] = {
    stage.name: stage for stage in CURRICULUM_STAGES
}

FULL_CURRICULUM_SEQUENCE: tuple[str, ...] = tuple(
    stage.name for stage in CURRICULUM_STAGES if stage.name != "recovery"
)


def resolve_curriculum_spec(
    spec: str,
    *,
    stage_override: str | int | None = None,
) -> tuple[CombatCurriculumStage, int, tuple[CombatCurriculumStage, ...]]:
    """Resolve curriculum CLI spec to (current_stage, stage_index, stage_sequence)."""
    normalized = spec.strip().lower()
    if normalized == "full":
        sequence = tuple(
            CURRICULUM_STAGE_BY_NAME[name] for name in FULL_CURRICULUM_SEQUENCE
        )
    elif normalized in CURRICULUM_STAGE_BY_NAME:
        stage = CURRICULUM_STAGE_BY_NAME[normalized]
        return stage, CURRICULUM_STAGES.index(stage), (stage,)
    else:
        valid = ", ".join(["full", *CURRICULUM_STAGE_BY_NAME.keys()])
        raise ValueError(f"Unknown curriculum {spec!r}. Valid: {valid}")

    if stage_override is None:
        return sequence[0], 0, sequence

    if isinstance(stage_override, int):
        if stage_override < 0 or stage_override >= len(sequence):
            raise ValueError(
                f"curriculum stage index {stage_override} out of range "
                f"[0, {len(sequence) - 1}]"
            )
        return sequence[stage_override], stage_override, sequence

    key = str(stage_override).strip().lower()
    for index, stage in enumerate(sequence):
        if stage.name == key:
            return stage, index, sequence
    raise ValueError(
        f"Unknown curriculum stage {stage_override!r} for spec {spec!r}"
    )


def parse_encounter_names(spec: str) -> list[EncounterSetup]:
    """Parse comma-separated encounter aliases into setup callables."""
    setups: list[EncounterSetup] = []
    for part in spec.split(","):
        key = part.strip().lower()
        if not key:
            continue
        setup = NAMED_ENCOUNTERS.get(key)
        if setup is None:
            valid = ", ".join(sorted(NAMED_ENCOUNTERS))
            raise ValueError(f"Unknown encounter {key!r}. Valid: {valid}")
        setups.append(setup)
    if not setups:
        raise ValueError(f"No encounters parsed from: {spec!r}")
    return setups


def parse_tier_flags(spec: str) -> dict[str, bool]:
    """Parse --include-tiers weak,normal,elite,boss into boolean flags."""
    tiers = {part.strip().lower() for part in spec.split(",") if part.strip()}
    if not tiers:
        raise ValueError("include-tiers must list at least one tier")
    valid = {"weak", "normal", "elite", "boss"}
    unknown = tiers - valid
    if unknown:
        raise ValueError(f"Unknown tiers: {unknown}. Valid: {valid}")
    return {
        "include_weak": "weak" in tiers,
        "include_normal": "normal" in tiers,
        "include_elite": "elite" in tiers,
        "include_boss": "boss" in tiers,
    }


def build_tier_encounter_pool(
    act_indices: tuple[int, ...],
    tier_flags: dict[str, bool],
    *,
    act1_biome: str = "random",
) -> list[EncounterSetup]:
    """Build encounter pool from tier flags (non-curriculum escape hatch)."""
    if act1_biome == "random" and 0 in act_indices:
        return build_mixed_act1_encounter_pool(act_indices, **tier_flags)
    return build_encounter_pool(act_indices, act1_biome=act1_biome, **tier_flags)


def _sample_hard_start(
    rng: np.random.Generator,
    stage: CombatCurriculumStage,
    character_ids: tuple[str, ...],
) -> EpisodeInitSample:
    char_id = resolve_character_for_episode(rng, character_ids)
    cfg = get_character(char_id)
    deck_template = stage.hard_start_deck_template
    deck = build_deck_template(deck_template, char_id)
    low, high = stage.hard_start_hp_range
    player_hp = int(rng.integers(low, high + 1))
    encounters = stage.hard_start_encounters or stage.encounter_pool
    enc_index = int(rng.integers(0, len(encounters)))
    encounter_setup = encounters[enc_index]
    return EpisodeInitSample(
        character_id=char_id,
        deck=deck,
        player_hp=player_hp,
        player_max_hp=cfg.starting_hp,
        encounter_setup=encounter_setup,
        is_hard_start=True,
        deck_template=deck_template,
    )


def sample_episode_init(
    rng: np.random.Generator,
    stage: CombatCurriculumStage,
    *,
    hard_start_fraction: float | None = None,
    character_ids: tuple[str, ...] | None = None,
) -> EpisodeInitSample:
    """Sample episode initialization for a curriculum stage."""
    chars = character_ids or stage.character_ids
    frac = stage.hard_start_fraction if hard_start_fraction is None else hard_start_fraction

    if frac > 0.0 and float(rng.random()) < frac:
        return _sample_hard_start(rng, stage, chars)

    char_id = resolve_character_for_episode(rng, chars)
    cfg = get_character(char_id)
    deck_template, deck = sample_deck_template(rng, stage.deck_templates, char_id)
    enc_index = int(rng.integers(0, len(stage.encounter_pool)))
    encounter_setup = stage.encounter_pool[enc_index]
    return EpisodeInitSample(
        character_id=char_id,
        deck=deck,
        player_hp=cfg.starting_hp,
        player_max_hp=cfg.starting_hp,
        encounter_setup=encounter_setup,
        is_hard_start=False,
        deck_template=deck_template,
    )


def encounter_setup_name(setup: EncounterSetup) -> str:
    """Stable encounter identifier for logging and info dicts."""
    return getattr(setup, "__name__", repr(setup))


def stage_at_index(index: int, sequence: Sequence[CombatCurriculumStage]) -> CombatCurriculumStage:
    if index < 0 or index >= len(sequence):
        raise IndexError(f"Stage index {index} out of range [0, {len(sequence) - 1}]")
    return sequence[index]

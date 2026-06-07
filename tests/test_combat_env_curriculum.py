"""Tests for curriculum-aware STS2CombatEnv initialization."""

from sts2_env.encounters import act1
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.training.combat_curriculum import (
    CURRICULUM_STAGE_BY_NAME,
    EpisodeInitSample,
)
from sts2_env.training.deck_templates import build_deck_template


def test_hard_start_reset_applies_compromised_state() -> None:
    deck = build_deck_template("stripped", "Ironclad")
    episode_init = EpisodeInitSample(
        character_id="Ironclad",
        deck=deck,
        player_hp=15,
        player_max_hp=80,
        encounter_setup=act1.setup_vantom_boss,
        is_hard_start=True,
        deck_template="stripped",
    )
    env = STS2CombatEnv(reward_shaping=False)
    _obs, info = env.reset(options={"episode_init": episode_init})

    assert env.combat is not None
    assert env.combat.player.current_hp == 15
    assert info["is_hard_start"] is True
    assert info["deck_template"] == "stripped"
    assert info["encounter_id"] == "setup_vantom_boss"
    assert len(env.combat.enemies) > 0


def test_curriculum_reset_uses_stage_pool() -> None:
    stage = CURRICULUM_STAGE_BY_NAME["easy_pair"]
    env = STS2CombatEnv(
        curriculum_stage=stage,
        character_ids=("Ironclad",),
        reward_shaping=False,
    )
    seen_encounters: set[str] = set()
    for seed in range(20):
        _obs, info = env.reset(seed=seed)
        seen_encounters.add(info["encounter_id"])
    assert seen_encounters.issubset(
        {"setup_fuzzy_wurm_crawler_weak", "setup_cultists_normal"}
    )


def test_step_info_includes_outcome_on_termination() -> None:
    stage = CURRICULUM_STAGE_BY_NAME["easy_pair"]
    env = STS2CombatEnv(
        curriculum_stage=stage,
        character_ids=("Ironclad",),
        reward_shaping=False,
        max_turns=1,
    )
    _obs, _info = env.reset(seed=0)
    _obs, _reward, terminated, truncated, info = env.step(0)
    if terminated or truncated:
        assert "won" in info
        assert "hp_remaining" in info
        assert "hp_ratio_end" in info

"""Tests for multi-character STS2CombatEnv."""

import pytest

from sts2_env.characters.all import (
    SUPPORTED_TRAINING_CHARACTERS,
    get_character,
    parse_character_ids,
    resolve_character_for_episode,
)
from sts2_env.gym_env.combat_env import STS2CombatEnv


@pytest.mark.parametrize("character_id", SUPPORTED_TRAINING_CHARACTERS)
def test_combat_env_reset_single_character(character_id: str):
    env = STS2CombatEnv(character_id=character_id)
    obs, info = env.reset(seed=42)
    char_cfg = get_character(character_id)

    assert obs.shape == env.observation_space.shape
    assert info["character_id"] == character_id
    assert env.combat is not None
    assert env.combat.character_id == character_id
    assert env.combat.player.max_hp == char_cfg.starting_hp
    total_cards = (
        len(env.combat.hand)
        + len(env.combat.draw_pile)
        + len(env.combat.discard_pile)
        + len(env.combat.exhaust_pile)
    )
    assert total_cards == char_cfg.starting_deck_size


def test_combat_env_defect_orb_queue():
    env = STS2CombatEnv(character_id="Defect")
    env.reset(seed=7)
    assert env.combat is not None
    assert env.combat.orb_queue is not None
    assert env.combat.orb_queue.capacity == 3


def test_combat_env_mixed_characters_covers_pool():
    character_ids = parse_character_ids("all")
    env = STS2CombatEnv(character_ids=character_ids)
    seen = set()
    for seed in range(50):
        _, info = env.reset(seed=seed)
        seen.add(info["character_id"])
    assert seen == set(character_ids)


def test_parse_character_ids_all():
    assert parse_character_ids("all") == SUPPORTED_TRAINING_CHARACTERS


def test_parse_character_ids_case_insensitive():
    assert parse_character_ids("ironclad,silent") == ("Ironclad", "Silent")


def test_resolve_character_for_episode():
    import numpy as np

    rng = np.random.default_rng(0)
    pool = ("Ironclad", "Silent")
    picked = resolve_character_for_episode(rng, pool)
    assert picked in pool

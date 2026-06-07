"""Tests for character-mechanics observation encoding."""

import pytest

from sts2_env.cards.defect import create_defect_starter_deck
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.cards.necrobinder import create_necrobinder_starter_deck
from sts2_env.cards.regent import create_regent_starter_deck
from sts2_env.characters.all import SUPPORTED_TRAINING_CHARACTERS, get_character
from sts2_env.core.combat import CombatState
from sts2_env.gym_env.observation import (
    BASE_OBS_SIZE,
    CHARACTER_MECHANICS_FEATURES,
    COMBAT_OBS_V2_SIZE,
    OBS_SIZE,
    POTION_OBS_SIZE,
    RELIC_OBS_SIZE,
    encode_observation,
)


def _make_combat(deck, character_id: str) -> CombatState:
    char_cfg = get_character(character_id)
    combat = CombatState(
        player_hp=char_cfg.starting_hp,
        player_max_hp=char_cfg.starting_hp,
        deck=deck,
        rng_seed=123,
        character_id=character_id,
        relics=[char_cfg.starting_relic],
    )
    combat.start_combat()
    return combat


def test_obs_v6_size_constants():
    assert COMBAT_OBS_V2_SIZE == BASE_OBS_SIZE + CHARACTER_MECHANICS_FEATURES
    assert OBS_SIZE == COMBAT_OBS_V2_SIZE + RELIC_OBS_SIZE + POTION_OBS_SIZE


def test_character_one_hot_ironclad():
    combat = _make_combat(create_ironclad_starter_deck(), "Ironclad")
    obs = encode_observation(combat)
    mechanics = obs[BASE_OBS_SIZE:]
    char_one_hot = mechanics[: len(SUPPORTED_TRAINING_CHARACTERS)]
    assert char_one_hot[0] == 1.0
    assert char_one_hot.sum() == 1.0


def test_stars_encoding_regent():
    combat = _make_combat(create_regent_starter_deck(), "Regent")
    combat.stars = 6
    obs = encode_observation(combat)
    stars_value = obs[BASE_OBS_SIZE + len(SUPPORTED_TRAINING_CHARACTERS)]
    assert stars_value == pytest.approx(6 / 30.0)


def test_orb_encoding_defect():
    combat = _make_combat(create_defect_starter_deck(), "Defect")
    combat.channel_orb(combat.player, "LIGHTNING")
    obs = encode_observation(combat)
    orb_count = obs[BASE_OBS_SIZE + len(SUPPORTED_TRAINING_CHARACTERS) + 1 + 1]
    assert orb_count > 0


def test_osty_encoding_necrobinder():
    combat = _make_combat(create_necrobinder_starter_deck(), "Necrobinder")
    combat.summon_osty(combat.player, 5)
    obs = encode_observation(combat)
    osty_start = BASE_OBS_SIZE + CHARACTER_MECHANICS_FEATURES - 3
    assert obs[osty_start] == 1.0
    assert obs[osty_start + 1] > 0

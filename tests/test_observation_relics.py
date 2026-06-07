"""Tests for relic observation encoding (obs v4)."""

import numpy as np
import pytest

from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.characters.all import get_character
from sts2_env.core.combat import CombatState
from sts2_env.gym_env.observation import (
    COMBAT_OBS_V2_SIZE,
    MAX_RELIC_SLOTS,
    OBS_SIZE,
    RELIC_FEATURES,
    RELIC_OBS_SIZE,
    TOKEN_SLICES,
    encode_observation,
    encode_relic_features,
    encode_relic_features_from_fields,
)
from sts2_env.relics.registry import create_relic_by_name


def _make_combat(relics: list[str] | None = None) -> CombatState:
    char_cfg = get_character("Ironclad")
    combat = CombatState(
        player_hp=char_cfg.starting_hp,
        player_max_hp=char_cfg.starting_hp,
        deck=create_ironclad_starter_deck(),
        rng_seed=42,
        character_id="Ironclad",
        relics=relics or [char_cfg.starting_relic],
    )
    combat.start_combat()
    return combat


def test_obs_v4_size_constants():
    assert OBS_SIZE == COMBAT_OBS_V2_SIZE + RELIC_OBS_SIZE
    assert OBS_SIZE == 294
    assert RELIC_OBS_SIZE == MAX_RELIC_SLOTS * RELIC_FEATURES
    assert TOKEN_SLICES["relics"] == (COMBAT_OBS_V2_SIZE, OBS_SIZE)


def test_relic_slots_populated_for_starter_relic():
    combat = _make_combat()
    obs = encode_observation(combat)
    assert obs.shape == (OBS_SIZE,)

    relic_slice = obs[COMBAT_OBS_V2_SIZE:OBS_SIZE]
    assert relic_slice[:RELIC_FEATURES].sum() > 0
    assert relic_slice[RELIC_FEATURES:].sum() == pytest.approx(0.0)


def test_empty_relic_slots_are_zero_padded():
    combat = _make_combat(relics=[])
    combat.relics = []
    obs = encode_observation(combat)
    assert obs[COMBAT_OBS_V2_SIZE:OBS_SIZE].sum() == pytest.approx(0.0)


def test_encode_relic_features_from_fields_matches_instance():
    relic = create_relic_by_name("BURNING_BLOOD")
    from_fields = encode_relic_features_from_fields(
        relic_id=relic.relic_id.name,
        rarity=relic.rarity.name,
        enabled=relic.enabled,
        is_used_up=relic.is_used_up,
    )
    from_instance = encode_relic_features(relic)
    np.testing.assert_allclose(from_fields, from_instance, rtol=1e-5)


def test_multiple_relics_fill_sequential_slots():
    combat = _make_combat(relics=["BURNING_BLOOD", "VAJRA", "ANCHOR"])
    obs = encode_observation(combat)
    relic_slice = obs[COMBAT_OBS_V2_SIZE:OBS_SIZE].reshape(MAX_RELIC_SLOTS, RELIC_FEATURES)
    filled = (np.abs(relic_slice).sum(axis=1) > 0).sum()
    assert filled == 3

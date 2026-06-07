"""Tests for potion observation encoding (obs v5)."""

import numpy as np
import pytest

from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.characters.all import get_character
from sts2_env.core.combat import CombatState
from sts2_env.gym_env.observation import (
    COMBAT_OBS_V2_SIZE,
    MAX_POTION_OBS_SLOTS,
    OBS_SIZE,
    POTION_FEATURES,
    POTION_OBS_SIZE,
    RELIC_OBS_SIZE,
    TOKEN_SLICES,
    encode_observation,
    encode_potion_features,
    encode_potion_features_from_fields,
)
from sts2_env.potions.base import create_potion


def _make_combat(potions: list | None = None) -> CombatState:
    char_cfg = get_character("Ironclad")
    combat = CombatState(
        player_hp=char_cfg.starting_hp,
        player_max_hp=char_cfg.starting_hp,
        deck=create_ironclad_starter_deck(),
        rng_seed=42,
        character_id="Ironclad",
        relics=[char_cfg.starting_relic],
        max_potion_slots=MAX_POTION_OBS_SLOTS,
    )
    combat.start_combat()
    if potions is not None:
        combat.potions = list(potions)
    return combat


def test_obs_v6_size_constants():
    assert OBS_SIZE == COMBAT_OBS_V2_SIZE + RELIC_OBS_SIZE + POTION_OBS_SIZE
    assert POTION_OBS_SIZE == MAX_POTION_OBS_SLOTS * POTION_FEATURES
    relic_end = COMBAT_OBS_V2_SIZE + RELIC_OBS_SIZE
    assert TOKEN_SLICES["potions"] == (relic_end, OBS_SIZE)
    assert TOKEN_SLICES["relics"] == (COMBAT_OBS_V2_SIZE, relic_end)


def test_potion_slots_populated():
    combat = _make_combat([
        create_potion("FirePotion"),
        create_potion("BlockPotion"),
        None,
    ])
    obs = encode_observation(combat)
    assert obs.shape == (OBS_SIZE,)

    potion_start, potion_end = TOKEN_SLICES["potions"]
    potion_slice = obs[potion_start:potion_end].reshape(MAX_POTION_OBS_SLOTS, POTION_FEATURES)
    assert potion_slice[0].sum() > 0
    assert potion_slice[1].sum() > 0
    assert potion_slice[2:].sum() == pytest.approx(0.0)


def test_empty_potion_slots_are_zero_padded():
    combat = _make_combat([None] * MAX_POTION_OBS_SLOTS)
    obs = encode_observation(combat)
    potion_start, potion_end = TOKEN_SLICES["potions"]
    assert obs[potion_start:potion_end].sum() == pytest.approx(0.0)


def test_encode_potion_features_from_fields_matches_instance():
    potion = create_potion("FirePotion")
    from_fields = encode_potion_features_from_fields(
        potion_id=potion.potion_id,
        rarity=potion.rarity,
        can_use_in_combat=potion.can_use_in_combat(),
    )
    from_instance = encode_potion_features(potion)
    np.testing.assert_allclose(from_fields, from_instance, rtol=1e-5)


def test_automatic_potion_has_zero_can_use_in_combat_flag():
    combat = _make_combat([create_potion("FairyInABottle")])
    obs = encode_observation(combat)
    potion_start, potion_end = TOKEN_SLICES["potions"]
    potion_slice = obs[potion_start:potion_end].reshape(MAX_POTION_OBS_SLOTS, POTION_FEATURES)
    assert potion_slice[0, 2] == pytest.approx(0.0)


def test_bridge_style_potion_fields_resolve_rarity_from_registry():
    potion = create_potion("FirePotion")
    from_fields = encode_potion_features_from_fields(
        potion_id="FirePotion",
        can_use_in_combat=True,
    )
    from_instance = encode_potion_features(potion)
    np.testing.assert_allclose(from_fields, from_instance, rtol=1e-5)

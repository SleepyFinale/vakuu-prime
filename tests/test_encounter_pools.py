"""Tests for act-mixed encounter pool construction."""

import pytest

from sts2_env.encounters.pools import (
    SUPPORTED_TRAINING_ACTS,
    build_encounter_pool,
    parse_act_indices,
)


class TestParseActIndices:
    def test_single_act(self):
        assert parse_act_indices("0") == (0,)

    def test_all_acts(self):
        assert parse_act_indices("all") == SUPPORTED_TRAINING_ACTS

    def test_mixed_list(self):
        assert parse_act_indices("0,1,2") == (0, 1, 2)


class TestBuildEncounterPool:
    def test_act0_non_empty_without_boss(self):
        pool = build_encounter_pool((0,), include_boss=False)
        assert len(pool) > 0

    def test_mixed_acts_larger_than_single(self):
        pool0 = build_encounter_pool((0,), include_boss=False)
        pool012 = build_encounter_pool((0, 1, 2), include_boss=False)
        assert len(pool012) > len(pool0)

    def test_exclude_boss_reduces_pool(self):
        with_boss = build_encounter_pool((0,), include_boss=True)
        without_boss = build_encounter_pool((0,), include_boss=False)
        assert len(with_boss) > len(without_boss)

    def test_empty_categories_raises(self):
        with pytest.raises(ValueError, match="Empty encounter pool"):
            build_encounter_pool(
                (0,),
                include_weak=False,
                include_normal=False,
                include_elite=False,
                include_boss=False,
            )

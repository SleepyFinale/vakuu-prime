"""Tests for 3-act runs with alternate Act 1 biomes (Overgrowth / Underdocks)."""

from __future__ import annotations

import pytest

from sts2_env.core.rng import Rng
from sts2_env.encounters.pools import encounter_lists_for_act
from sts2_env.encounters.registry import get_boss_setup
from sts2_env.map.acts import (
    ACT_1_OVERGROWTH,
    ACT_1_UNDERDOCKS,
    build_run_acts,
)
from sts2_env.run.run_state import RunState


class TestBuildRunActs:
    def test_forced_overgrowth(self):
        rng = Rng(1)
        acts = build_run_acts(rng, act1_biome="overgrowth")
        assert len(acts) == 3
        assert acts[0].biome_id == "overgrowth"
        assert acts[1].biome_id == "hive"
        assert acts[2].biome_id == "glory"

    def test_forced_underdocks(self):
        rng = Rng(1)
        acts = build_run_acts(rng, act1_biome="underdocks")
        assert acts[0].biome_id == "underdocks"
        assert acts[0].boss_ids == ACT_1_UNDERDOCKS.boss_ids

    def test_locked_underdocks_stays_overgrowth(self):
        rng = Rng(99)
        acts = build_run_acts(
            rng,
            underdocks_unlocked=False,
            act1_biome="random",
        )
        assert acts[0].biome_id == "overgrowth"

    def test_undiscovered_forces_underdocks(self):
        rng = Rng(99)
        acts = build_run_acts(
            rng,
            underdocks_discovered=False,
            act1_biome="random",
        )
        assert acts[0].biome_id == "underdocks"

    def test_random_respects_rng(self):
        rng_a = Rng(42, "act_list")
        rng_b = Rng(42, "act_list")
        assert build_run_acts(rng_a, act1_biome="random")[0].biome_id == build_run_acts(
            rng_b, act1_biome="random"
        )[0].biome_id

    def test_underdocks_override_requires_unlock(self):
        with pytest.raises(ValueError, match="underdocks_unlocked"):
            build_run_acts(Rng(0), act1_biome="underdocks", underdocks_unlocked=False)


class TestEncounterRouting:
    def test_act0_overgrowth_uses_act1_pool(self):
        from sts2_env.encounters import act1

        weak, *_ = encounter_lists_for_act(0, biome_id="overgrowth")
        assert weak == list(act1.WEAK_ENCOUNTERS)

    def test_act0_underdocks_uses_act4_pool(self):
        from sts2_env.encounters import act4

        weak, *_ = encounter_lists_for_act(0, biome_id="underdocks")
        assert weak == list(act4.WEAK_ENCOUNTERS)

    def test_underdocks_bosses_registered(self):
        for boss_id in ACT_1_UNDERDOCKS.boss_ids:
            assert get_boss_setup(boss_id) is not None


class TestRunStateAct1Biome:
    def test_run_state_underdocks_act1(self):
        rs = RunState(seed=7, act1_biome="underdocks")
        assert rs.acts[0].biome_id == "underdocks"
        assert rs.current_act.biome_id == "underdocks"

    def test_run_state_three_acts_only(self):
        rs = RunState(seed=7, act1_biome="overgrowth")
        assert len(rs.acts) == 3
        assert rs.acts[0].biome_id == "overgrowth"

    def test_overgrowth_and_underdocks_differ_events(self):
        og = ACT_1_OVERGROWTH.act_event_ids
        ud = ACT_1_UNDERDOCKS.act_event_ids
        assert og != ud
        assert "AromaOfChaos" in og
        assert "AbyssalBaths" in ud

"""Bridge combat hydration round-trip tests."""

from __future__ import annotations

import numpy as np

from sts2_env.bridge.combat_hydration import (
    combat_to_full_bridge_state,
    hydrate_combat_from_bridge,
)
from sts2_env.gym_env.observation import encode_observation


def test_sim_bridge_round_trip_observation(simple_combat):
    payload = combat_to_full_bridge_state(simple_combat)
    result = hydrate_combat_from_bridge(payload)
    assert result.ok, result.warnings

    obs_src = encode_observation(simple_combat)
    obs_hyd = encode_observation(result.combat)
    assert np.allclose(obs_src, obs_hyd, atol=0.05)


def test_hydration_fails_without_player():
    result = hydrate_combat_from_bridge({"type": "combat_action"})
    assert not result.ok

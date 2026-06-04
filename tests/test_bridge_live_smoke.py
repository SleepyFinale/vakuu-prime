"""Bridge live-smoke parity tests.

Two layers:

1. An offline golden compare that runs in CI: the committed
   ``smoke_combat.json`` fixture is replayed against the simulator. This guards
   the serialization/comparison harness and the smoke scenario without needing
   the game.
2. A live test (``@pytest.mark.live_bridge``, skipped unless
   ``--run-live-bridge``) that connects to a running STS2 bridge, records a few
   combat steps, and compares them against the simulator.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import sts2_env.powers  # noqa: F401

from sts2_env.parity.bridge_replay import (
    BridgeReplayRecorder,
    compare_combat_replay,
    load_replay_trace,
)
from sts2_env.parity.bridge_smoke import (
    SMOKE_COMBAT_ACTIONS,
    SMOKE_FACTORY,
    build_smoke_combat_trace,
    make_smoke_combat,
)

FIXTURE = Path(__file__).parent / "fixtures" / "bridge_replays" / "smoke_combat.json"


def test_smoke_combat_fixture_matches_simulator():
    """Committed golden trace replays cleanly against the simulator (offline CI)."""
    result = compare_combat_replay(str(FIXTURE), factory=SMOKE_FACTORY)
    assert result.success, "\n".join(result.mismatches)


def test_smoke_fixture_is_in_sync_with_generator():
    """Fixture on disk equals the freshly generated trace (regenerate if this fails)."""
    fresh = build_smoke_combat_trace().to_dict()
    on_disk = load_replay_trace(FIXTURE).to_dict()
    assert on_disk == fresh, (
        "smoke_combat.json is stale; regenerate with "
        "`python scripts/record_bridge_smoke.py`"
    )


@pytest.mark.live_bridge
def test_live_combat_matches_simulator():
    """Record a few live combat steps and compare against the simulator.

    Assumes a running STS2 game with the bridge mod, already inside a combat
    that matches :func:`make_smoke_combat` (Ironclad Act 1 vs Shrinker Beetle).
    """
    from sts2_env.bridge.client import STS2GameClient

    host = os.environ.get("STS2_BRIDGE_HOST", "127.0.0.1")
    port = int(os.environ.get("STS2_BRIDGE_PORT", "9002"))

    with STS2GameClient(host=host, port=port) as raw_client:
        recorder = BridgeReplayRecorder(
            raw_client, metadata={"scenario_factory": SMOKE_FACTORY, "source": "live"}
        )
        recorder.receive_state()
        for action in SMOKE_COMBAT_ACTIONS:
            recorder.send_action(dict(action))
            recorder.receive_state()

    result = compare_combat_replay(recorder.trace, factory=SMOKE_FACTORY)
    assert result.success, "\n".join(result.mismatches)

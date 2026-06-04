#!/usr/bin/env python3
"""Record the bridge smoke trace.

Default (offline) mode regenerates the committed golden fixture by stepping the
deterministic simulator scenario. ``--live`` connects to a running STS2 bridge,
records the scripted smoke actions, saves the trace, and compares it against the
simulator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sts2_env.parity.bridge_replay import (
    BridgeReplayRecorder,
    compare_combat_replay,
    save_replay_trace,
)
from sts2_env.parity.bridge_smoke import (
    SMOKE_COMBAT_ACTIONS,
    SMOKE_FACTORY,
    build_smoke_combat_trace,
)

DEFAULT_FIXTURE = REPO_ROOT / "tests/fixtures/bridge_replays/smoke_combat.json"


def _record_offline(out: Path) -> int:
    trace = build_smoke_combat_trace()
    out.parent.mkdir(parents=True, exist_ok=True)
    save_replay_trace(trace, out)
    result = compare_combat_replay(str(out), factory=SMOKE_FACTORY)
    print(f"wrote {out} ({len(trace.steps)} steps); self-compare ok={result.success}")
    return 0 if result.success else 1


def _record_live(out: Path, host: str, port: int) -> int:
    from sts2_env.bridge.client import STS2GameClient

    with STS2GameClient(host=host, port=port) as raw_client:
        recorder = BridgeReplayRecorder(
            raw_client, metadata={"scenario_factory": SMOKE_FACTORY, "source": "live"}
        )
        recorder.receive_state()
        for action in SMOKE_COMBAT_ACTIONS:
            recorder.send_action(dict(action))
            recorder.receive_state()
        save_replay_trace(recorder.trace, out)

    result = compare_combat_replay(str(out), factory=SMOKE_FACTORY)
    print(f"recorded live trace to {out}; compare ok={result.success}")
    for mismatch in result.mismatches:
        print(f"  {mismatch}")
    return 0 if result.success else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_FIXTURE), help="Output trace path.")
    parser.add_argument("--live", action="store_true", help="Record from a running STS2 bridge.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9002)
    args = parser.parse_args()

    out = Path(args.out)
    if args.live:
        return _record_live(out, args.host, args.port)
    return _record_offline(out)


if __name__ == "__main__":
    raise SystemExit(main())

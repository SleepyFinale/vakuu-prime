# Bridge Live Smoke

This is the end-to-end smoke procedure that validates the simulator against the
real Slay the Spire 2 game through the TCP bridge mod. It complements the
offline-only [`BRIDGE_REPLAY_HARNESS.md`](BRIDGE_REPLAY_HARNESS.md) by adding a
build gate and a live record-and-compare step.

There are two layers:

- **Offline (runs in CI, no game required).** A committed golden trace
  ([`tests/fixtures/bridge_replays/smoke_combat.json`](../tests/fixtures/bridge_replays/smoke_combat.json))
  is replayed against the simulator. This guards the serialization and
  comparison harness plus the deterministic smoke scenario.
- **Live (manual, requires the game).** The agent connects to the running game,
  records a few combat steps, and compares them against the simulator.

## Components

- Build gate: [`scripts/bridge_live_smoke.ps1`](../scripts/bridge_live_smoke.ps1)
- Deterministic scenario + golden builder: [`sts2_env/parity/bridge_smoke.py`](../sts2_env/parity/bridge_smoke.py)
- Recorder helper: [`scripts/record_bridge_smoke.py`](../scripts/record_bridge_smoke.py)
- Tests: [`tests/test_bridge_live_smoke.py`](../tests/test_bridge_live_smoke.py)
- Recorder + comparator: [`sts2_env/parity/bridge_replay.py`](../sts2_env/parity/bridge_replay.py)

## The smoke scenario

[`make_smoke_combat`](../sts2_env/parity/bridge_smoke.py) builds an Ironclad Act 1
combat against a Shrinker Beetle with a fixed hand (`Strike`, `Defend`, `Bash`)
and 3 energy. The scripted actions are: play `Strike` at the enemy, play
`Defend`, then end the turn. Because the factory is fully deterministic, the
simulator produces a stable golden trace.

The smoke scenario remains **Ironclad-only**; multi-character bridge support
does not change this golden trace. To evaluate other characters live, set
`STS2_BRIDGE_CHARACTER` before launching the game (see below) and use a
character-matched combat model in the agent runner.

## Character selection (live bridge)

The bridge mod reads `STS2_BRIDGE_CHARACTER` at startup (default `Ironclad`).
Valid values (case-insensitive): `Ironclad`, `Silent`, `Defect`, `Regent`,
`Necrobinder`. The mod selects that character on the main menu before the agent
connects; every combat state JSON includes `player.character_id` plus mechanics
fields (`stars`, `orb_queue`, `osty`), `relics` (with `counter`), `potions`, pile card
arrays (`draw_pile`, `discard_pile`, `play_pile`), and enriched hand fields (`keywords`,
`base_damage`, `base_block`, `retain`, `hit_count`) for the **1985-dim obs v11** encoder.

```powershell
$env:STS2_BRIDGE_CHARACTER = "Silent"
# Launch STS2 with the bridge mod, then run agent_runner with a Silent-trained model
```

Match the agent's `--character` or `--combat-models-by-character` path to the
character the mod selected.

## Offline (CI) usage

```powershell
# Regenerate the golden fixture and self-compare.
python scripts/record_bridge_smoke.py

# Offline parity tests (golden compare + harness).
python -m pytest tests/test_bridge_live_smoke.py tests/test_bridge_replay_harness.py -q
```

The offline golden compare is also run as part of the build gate below.

## Build gate

```powershell
./scripts/bridge_live_smoke.ps1
```

This script:

1. Builds the C# bridge mod (`dotnet build bridge_mod/STS2BridgeMod.csproj -c Release`).
   It fails fast on missing `dotnet` and on any C# compile error. If the C#
   compiles but the Godot `.pck` export is skipped because the Godot 4.5.1 mono
   editor is not configured (see `GodotPath` in
   [`bridge_mod/STS2BridgeMod.csproj`](../bridge_mod/STS2BridgeMod.csproj)), it
   warns and continues — the loadable `.pck` is only needed to run the mod
   in-game, not to validate that the bridge code compiles.
2. Regenerates the offline golden fixture and self-compares.
3. Runs the offline bridge parity tests.

## Live record + compare (manual)

Prerequisites: the Godot 4.5.1 mono editor configured so the build produces a
loadable `.pck`, and Slay the Spire 2 installed with the bridge mod in its
`mods/` folder.

1. Build and install the mod (with Godot configured):

```powershell
./scripts/bridge_live_smoke.ps1
```

2. Launch Slay the Spire 2 with the bridge mod loaded and start an Ironclad Act
   1 combat that matches the smoke scenario (vs a Shrinker Beetle).

3. With the game waiting on the combat state, record and compare:

```powershell
python scripts/record_bridge_smoke.py --live --host 127.0.0.1 --port 9002
```

   This connects through [`STS2GameClient`](../sts2_env/bridge/client.py), records
   the scripted smoke actions, writes the trace, and runs
   `compare_combat_replay` against the simulator. A clean run prints
   `compare ok=True`.

4. Optionally drive the live pytest directly (skipped unless the flag is given):

```powershell
python -m pytest tests/test_bridge_live_smoke.py --run-live-bridge -q
```

For longer slices (map → combat → reward → rest), use the agent runner's replay
recorder as described in [`BRIDGE_REPLAY_HARNESS.md`](BRIDGE_REPLAY_HARNESS.md):

```powershell
python -m sts2_env.bridge.agent_runner --record-replay artifacts/smoke_run.json --replay-factory <module:function>
python -m sts2_env.parity.bridge_replay_cli compare artifacts/smoke_run.json --mode run --factory <module:function>
```

## Current status

- Build gate, offline golden fixture, and offline parity tests are validated and
  run without the game (last run: 2026-06-04, `bridge_live_smoke: OK`, 31 pytest
  passes + golden self-compare).
- C# `dotnet build` succeeds; `.pck` export still requires the Godot 4.5.1 mono
  editor path in `bridge_mod/STS2BridgeMod.csproj`.
- The live record-and-compare step has **not** been recorded in CI: it requires
  the game running with the mod loaded. Run manually on a machine that has STS2
  installed:

  `python scripts/record_bridge_smoke.py --live --host 127.0.0.1 --port 9002`

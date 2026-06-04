"""Deterministic smoke scenario for the live-game bridge parity harness.

The scenario is intentionally tiny: an Ironclad Act 1 combat with a fixed hand
and a short scripted action sequence. Because the factory is fully
deterministic, stepping the simulator produces a stable golden trace that CI can
replay offline (``compare_combat_replay``) without the game running, and that a
live recording can be diffed against.
"""

from __future__ import annotations

from sts2_env.bridge.protocol import BridgeAction
from sts2_env.cards.ironclad_basic import (
    create_ironclad_starter_deck,
    make_bash,
    make_defend_ironclad,
    make_strike_ironclad,
)
from sts2_env.core.combat import CombatState
from sts2_env.core.rng import Rng
from sts2_env.monsters.act1_weak import create_shrinker_beetle
from sts2_env.parity.bridge_replay import (
    BridgeReplayStep,
    BridgeReplayTrace,
    combat_state_to_bridge_state,
)

SMOKE_SEED = 1234
SMOKE_FACTORY = "sts2_env.parity.bridge_smoke:make_smoke_combat"

# Strike (target the beetle), Defend (self block), then end the turn.
SMOKE_COMBAT_ACTIONS: tuple[dict[str, object], ...] = (
    {"action": BridgeAction.PLAY, "card_index": 0, "target_index": 0},
    {"action": BridgeAction.PLAY, "card_index": 0, "target_index": -1},
    {"action": BridgeAction.END_TURN},
)


def make_smoke_combat() -> CombatState:
    """Build the deterministic Ironclad smoke combat."""
    combat = CombatState(
        player_hp=80,
        player_max_hp=80,
        deck=create_ironclad_starter_deck(),
        rng_seed=SMOKE_SEED,
        character_id="Ironclad",
    )
    creature, ai = create_shrinker_beetle(Rng(SMOKE_SEED))
    combat.add_enemy(creature, ai)
    combat.start_combat()
    # Fixed hand so the scenario does not depend on opening-draw order.
    combat.hand = [make_strike_ironclad(), make_defend_ironclad(), make_bash()]
    combat.energy = 3
    return combat


def build_smoke_combat_trace() -> BridgeReplayTrace:
    """Step the simulator to produce a golden combat trace for offline compare."""
    from sts2_env.parity.bridge_replay import _apply_replay_action

    combat = make_smoke_combat()
    trace = BridgeReplayTrace(
        mode="combat",
        metadata={"scenario_factory": SMOKE_FACTORY, "source": "simulator"},
        initial_state=combat_state_to_bridge_state(combat),
    )
    for action in SMOKE_COMBAT_ACTIONS:
        _apply_replay_action(combat, dict(action))
        trace.steps.append(
            BridgeReplayStep(
                action=dict(action),
                resulting_state=combat_state_to_bridge_state(combat),
            )
        )
    return trace

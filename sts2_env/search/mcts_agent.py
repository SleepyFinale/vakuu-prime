"""High-level combat action selection with optional MCTS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sts2_env.core.combat import CombatState
from sts2_env.gym_env.action_space import get_action_mask
from sts2_env.gym_env.observation import encode_observation
from sts2_env.search.mcts_combat import MCTSConfig, MCTSStats, mcts_search

__all__ = ["MCTSConfig", "MCTSStats", "build_mcts_config", "select_combat_action"]


def build_mcts_config(args: Any) -> MCTSConfig | None:
    """Build ``MCTSConfig`` from CLI args with ``--mcts`` / ``--mcts-sims`` fields."""
    if not getattr(args, "mcts", False):
        return None
    time_budget = getattr(args, "mcts_time_budget", None)
    return MCTSConfig(
        n_simulations=int(getattr(args, "mcts_sims", 128)),
        c_puct=float(getattr(args, "mcts_c_puct", 1.5)),
        max_actions_per_turn=int(getattr(args, "mcts_max_depth", 30)),
        time_budget_s=time_budget,
    )


def select_combat_action(
    combat: CombatState,
    model: Any,
    *,
    mcts_config: MCTSConfig | None = None,
    stats: MCTSStats | None = None,
) -> int:
    """Choose a combat action with optional turn-bounded MCTS."""
    if mcts_config is None:
        obs = encode_observation(combat)
        mask = get_action_mask(combat)
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
    return mcts_search(combat, model, mcts_config, stats=stats)

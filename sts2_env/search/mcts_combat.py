"""Turn-bounded PUCT MCTS for combat inference."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sts2_env.core.combat import CombatState
from sts2_env.core.constants import ACTION_END_TURN
from sts2_env.gym_env.action_space import get_action_mask
from sts2_env.gym_env.combat_value import predict_combat_values
from sts2_env.gym_env.observation import encode_observation
from sts2_env.search.combat_clone import clone_combat_state
from sts2_env.search.combat_step import apply_combat_action
from sts2_env.search.policy_guide import policy_prior_and_value


@dataclass(frozen=True)
class MCTSConfig:
    n_simulations: int = 128
    c_puct: float = 1.5
    max_actions_per_turn: int = 30
    temperature: float = 0.0
    leaf_eval: str = "post_enemy_critic"
    time_budget_s: float | None = None


@dataclass
class MCTSStats:
    root_visits: dict[int, int] = field(default_factory=dict)
    root_values: dict[int, float] = field(default_factory=dict)
    root_priors: dict[int, float] = field(default_factory=dict)
    simulations_run: int = 0
    elapsed_s: float = 0.0


class _MCTSNode:
    __slots__ = (
        "combat",
        "parent",
        "action",
        "children",
        "visit_count",
        "total_value",
        "prior",
        "expanded",
        "terminal",
        "cached_value",
        "depth",
    )

    def __init__(
        self,
        combat: CombatState,
        *,
        parent: _MCTSNode | None = None,
        action: int | None = None,
        prior: float = 0.0,
        depth: int = 0,
        terminal: bool = False,
        cached_value: float | None = None,
    ):
        self.combat = combat
        self.parent = parent
        self.action = action
        self.children: dict[int, _MCTSNode] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.expanded = terminal
        self.terminal = terminal
        self.cached_value = cached_value
        self.depth = depth

    @property
    def q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


def _terminal_value(combat: CombatState, model: Any) -> float:
    if combat.is_over:
        return 1.0 if combat.player_won else -1.0
    obs = encode_observation(combat)
    return float(predict_combat_values(model, obs)[0])


def _is_search_terminal(combat: CombatState, action: int | None) -> bool:
    if combat.is_over:
        return True
    if action == ACTION_END_TURN:
        return True
    return False


def _select_child(node: _MCTSNode, c_puct: float) -> _MCTSNode:
    total_visits = sum(child.visit_count for child in node.children.values())
    sqrt_total = math.sqrt(max(total_visits, 1))
    best_score = -float("inf")
    best_child: _MCTSNode | None = None
    for action, child in node.children.items():
        if child.visit_count == 0:
            q = 0.0
        else:
            q = child.q
        u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_child = child
    assert best_child is not None
    return best_child


def _expand(node: _MCTSNode, model: Any, config: MCTSConfig) -> float:
    if node.terminal:
        if node.cached_value is not None:
            return node.cached_value
        value = _terminal_value(node.combat, model)
        node.cached_value = value
        return value

    mask = get_action_mask(node.combat)
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        node.expanded = True
        node.terminal = True
        value = _terminal_value(node.combat, model)
        node.cached_value = value
        return value

    obs = encode_observation(node.combat)
    priors, value = policy_prior_and_value(model, obs, mask)

    for action in legal:
        action_int = int(action)
        child_combat = clone_combat_state(node.combat)
        apply_combat_action(child_combat, action_int)
        terminal = _is_search_terminal(child_combat, action_int)
        cached = _terminal_value(child_combat, model) if terminal else None
        child = _MCTSNode(
            child_combat,
            parent=node,
            action=action_int,
            prior=float(priors[action_int]),
            depth=node.depth + 1,
            terminal=terminal,
            cached_value=cached,
        )
        node.children[action_int] = child

    node.expanded = True
    return value


def _simulate(root: _MCTSNode, model: Any, config: MCTSConfig) -> float:
    node = root
    path: list[_MCTSNode] = [node]

    while node.expanded and not node.terminal and node.children:
        if node.depth >= config.max_actions_per_turn:
            break
        node = _select_child(node, config.c_puct)
        path.append(node)

    if node.terminal:
        leaf_value = node.cached_value if node.cached_value is not None else _terminal_value(node.combat, model)
    elif not node.expanded:
        leaf_value = _expand(node, model, config)
    elif node.depth >= config.max_actions_per_turn:
        leaf_value = _terminal_value(node.combat, model)
    else:
        leaf_value = _terminal_value(node.combat, model)

    for path_node in reversed(path):
        path_node.visit_count += 1
        path_node.total_value += leaf_value

    return leaf_value


def mcts_search(
    combat: CombatState,
    model: Any,
    config: MCTSConfig | None = None,
    *,
    stats: MCTSStats | None = None,
) -> int:
    """Run turn-bounded MCTS and return the best root action index."""
    config = config or MCTSConfig()
    root = _MCTSNode(combat, depth=0)
    start = time.perf_counter()
    sims = 0

    while sims < config.n_simulations:
        if config.time_budget_s is not None and (time.perf_counter() - start) >= config.time_budget_s:
            break
        _simulate(root, model, config)
        sims += 1

    elapsed = time.perf_counter() - start

    if not root.children:
        mask = get_action_mask(combat)
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            return ACTION_END_TURN
        obs = encode_observation(combat)
        priors, _ = policy_prior_and_value(model, obs, mask)
        return int(legal[np.argmax(priors[legal])])

    if config.temperature <= 0:
        best_action = max(root.children, key=lambda a: root.children[a].visit_count)
    else:
        visits = np.array([root.children[a].visit_count for a in root.children], dtype=np.float64)
        actions = list(root.children.keys())
        probs = visits ** (1.0 / config.temperature)
        probs /= probs.sum()
        best_action = int(np.random.choice(actions, p=probs))

    if stats is not None:
        stats.simulations_run = sims
        stats.elapsed_s = elapsed
        for action, child in root.children.items():
            stats.root_visits[action] = child.visit_count
            stats.root_values[action] = child.q
            stats.root_priors[action] = child.prior

    return int(best_action)

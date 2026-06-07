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
from sts2_env.search.combat_step import apply_combat_action_for_search
from sts2_env.search.policy_guide import policy_prior_and_value


@dataclass(frozen=True)
class MCTSConfig:
    n_simulations: int = 128
    c_puct: float = 1.5
    max_actions_per_turn: int = 15
    lookahead_player_turns: int = 1
    temperature: float = 0.0
    leaf_eval: str = "post_enemy_critic"
    time_budget_s: float | None = None
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25


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
        "fully_expanded",
        "terminal",
        "cached_value",
        "cached_priors",
        "cached_legal",
        "cached_expand_value",
        "player_turn_index",
        "actions_this_turn",
    )

    def __init__(
        self,
        combat: CombatState,
        *,
        parent: _MCTSNode | None = None,
        action: int | None = None,
        prior: float = 0.0,
        player_turn_index: int = 0,
        actions_this_turn: int = 0,
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
        self.fully_expanded = terminal
        self.terminal = terminal
        self.cached_value = cached_value
        self.cached_priors: dict[int, float] | None = None
        self.cached_legal: np.ndarray | None = None
        self.cached_expand_value: float | None = None
        self.player_turn_index = player_turn_index
        self.actions_this_turn = actions_this_turn

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


def _is_search_terminal(combat: CombatState) -> bool:
    return combat.is_over or combat.pending_choice is not None


def _advance_after_player_end(
    combat: CombatState,
    *,
    player_turn_index: int,
    config: MCTSConfig,
    model: Any,
) -> tuple[int, int, bool, float | None]:
    """Run enemy phase after player END_TURN; return (turn_index, actions, terminal, cached)."""
    if combat.is_over:
        return player_turn_index, 0, True, None

    if player_turn_index < config.lookahead_player_turns:
        combat.advance_enemy_phase(resume_player_turn=True)
        if _is_search_terminal(combat):
            return player_turn_index, 0, True, None
        return player_turn_index + 1, 0, False, None

    if player_turn_index == 0 and config.lookahead_player_turns == 0:
        combat.advance_enemy_phase(resume_player_turn=False)
        return player_turn_index, 0, True, _terminal_value(combat, model)

    # Lookahead player turn ended — score without a second enemy phase.
    return player_turn_index, 0, True, _terminal_value(combat, model)


def _apply_search_action(
    combat: CombatState,
    action: int,
    *,
    player_turn_index: int,
    actions_this_turn: int,
    config: MCTSConfig,
    model: Any,
) -> tuple[int, int, bool, float | None]:
    """Apply one search action and return (turn_index, actions, terminal, cached_value)."""
    if action == ACTION_END_TURN:
        extra_turn = combat.finish_player_turn_only()
        if combat.is_over:
            return player_turn_index, 0, True, None
        if extra_turn:
            return player_turn_index, 0, False, None
        turn_index, actions, terminal, cached = _advance_after_player_end(
            combat,
            player_turn_index=player_turn_index,
            config=config,
            model=model,
        )
        return turn_index, actions, terminal, cached

    apply_combat_action_for_search(combat, action)
    if _is_search_terminal(combat):
        return player_turn_index, actions_this_turn + 1, True, None
    return player_turn_index, actions_this_turn + 1, False, None


def _rollout_to_post_enemy(combat: CombatState, model: Any, config: MCTSConfig) -> float:
    """Fast-forward a mid-turn snapshot through END_TURN and one enemy phase."""
    work = clone_combat_state(combat)
    max_steps = config.max_actions_per_turn * 2
    for _ in range(max_steps):
        if _is_search_terminal(work):
            break
        mask = get_action_mask(work)
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            break
        if ACTION_END_TURN in legal:
            work.finish_player_turn_only()
            if work.is_over:
                break
            work.advance_enemy_phase(resume_player_turn=False)
            break
        obs = encode_observation(work)
        priors, _ = policy_prior_and_value(model, obs, mask)
        action = int(legal[np.argmax(priors[legal])])
        apply_combat_action_for_search(work, action)

    return _terminal_value(work, model)


def _leaf_value(node: _MCTSNode, model: Any, config: MCTSConfig) -> float:
    if node.cached_value is not None:
        return node.cached_value
    if (
        config.leaf_eval == "post_enemy_critic"
        and node.player_turn_index == 0
        and not node.terminal
    ):
        return _rollout_to_post_enemy(node.combat, model, config)
    return _terminal_value(node.combat, model)


def _apply_root_dirichlet_noise(
    priors: dict[int, float],
    config: MCTSConfig,
) -> None:
    if config.dirichlet_epsilon <= 0.0 or len(priors) <= 1:
        return
    actions = list(priors.keys())
    policy = np.array([priors[a] for a in actions], dtype=np.float64)
    noise = np.random.dirichlet([config.dirichlet_alpha] * len(actions))
    mixed = (1.0 - config.dirichlet_epsilon) * policy + config.dirichlet_epsilon * noise
    for action, prior in zip(actions, mixed):
        priors[action] = float(prior)


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


def _expand_one(node: _MCTSNode, model: Any, config: MCTSConfig) -> float:
    if node.terminal:
        if node.cached_value is not None:
            return node.cached_value
        value = _terminal_value(node.combat, model)
        node.cached_value = value
        return value

    if node.cached_legal is None:
        mask = get_action_mask(node.combat)
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            node.expanded = True
            node.fully_expanded = True
            node.terminal = True
            value = _leaf_value(node, model, config)
            node.cached_value = value
            return value

        obs = encode_observation(node.combat)
        priors, value = policy_prior_and_value(model, obs, mask)
        node.cached_legal = legal
        node.cached_priors = {int(action): float(priors[int(action)]) for action in legal}
        if node.parent is None:
            _apply_root_dirichlet_noise(node.cached_priors, config)
        node.cached_expand_value = value

    legal = node.cached_legal
    assert legal is not None
    assert node.cached_priors is not None
    assert node.cached_expand_value is not None

    unexpanded = [int(action) for action in legal if int(action) not in node.children]
    if not unexpanded:
        node.expanded = True
        node.fully_expanded = True
        return node.cached_expand_value

    action_int = max(unexpanded, key=lambda action: (node.cached_priors[action], action))
    child_combat = clone_combat_state(node.combat)
    turn_index, actions, terminal, cached = _apply_search_action(
        child_combat,
        action_int,
        player_turn_index=node.player_turn_index,
        actions_this_turn=node.actions_this_turn,
        config=config,
        model=model,
    )
    if terminal and cached is None:
        cached = _terminal_value(child_combat, model)
    child = _MCTSNode(
        child_combat,
        parent=node,
        action=action_int,
        prior=node.cached_priors[action_int],
        player_turn_index=turn_index,
        actions_this_turn=actions,
        terminal=terminal,
        cached_value=cached,
    )
    node.children[action_int] = child
    node.expanded = True
    if len(node.children) >= len(legal):
        node.fully_expanded = True
    return node.cached_expand_value


def _expand_all_for_tests(node: _MCTSNode, model: Any, config: MCTSConfig) -> float:
    """Expand every legal child; used by tests that need a fully materialized node."""
    value = _expand_one(node, model, config)
    while not node.fully_expanded and not node.terminal:
        value = _expand_one(node, model, config)
    return value


def _simulate(root: _MCTSNode, model: Any, config: MCTSConfig) -> float:
    node = root
    path: list[_MCTSNode] = [node]

    while node.fully_expanded and not node.terminal and node.children:
        if node.actions_this_turn >= config.max_actions_per_turn:
            break
        node = _select_child(node, config.c_puct)
        path.append(node)

    if node.terminal:
        leaf_value = node.cached_value if node.cached_value is not None else _terminal_value(node.combat, model)
    elif not node.fully_expanded:
        leaf_value = _expand_one(node, model, config)
    elif node.actions_this_turn >= config.max_actions_per_turn:
        leaf_value = _leaf_value(node, model, config)
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
    root = _MCTSNode(combat, player_turn_index=0, actions_this_turn=0)
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
        best_action = max(
            root.children,
            key=lambda a: (
                root.children[a].visit_count,
                root.children[a].q,
                0 if a != ACTION_END_TURN else -1,
            ),
        )
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

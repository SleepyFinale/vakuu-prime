"""MCTS combat sequencing tests."""

from __future__ import annotations

import numpy as np

from sts2_env.cards.base import reset_instance_counter
from sts2_env.cards.ironclad import make_inflame
from sts2_env.cards.ironclad_basic import make_defend_ironclad, make_strike_ironclad
from sts2_env.core.combat import CombatState
from sts2_env.core.constants import ACTION_END_TURN
from sts2_env.core.enums import CombatSide, PowerId
from sts2_env.core.rng import Rng
from sts2_env.gym_env.action_space import get_action_mask
from sts2_env.gym_env.observation import _POWER_ID_TO_PLAYER_IDX
from sts2_env.monsters.act1_weak import create_shrinker_beetle
from sts2_env.search.combat_clone import clone_combat_state
from sts2_env.search.mcts_combat import (
    MCTSConfig,
    MCTSStats,
    _MCTSNode,
    _apply_root_dirichlet_noise,
    _apply_search_action,
    _expand_all_for_tests,
    mcts_search,
)


class _MockDistribution:
    def __init__(self, probs: np.ndarray):
        import torch

        self.distribution = type("Dist", (), {})()
        self.distribution.probs = torch.as_tensor(probs, dtype=torch.float32).unsqueeze(0)


class _MockPolicy:
    """Policy with uniform priors; ``predict`` still prefers the strike action."""

    def __init__(self, *, value_from_hp: bool = False):
        self._value_from_hp = value_from_hp

    def get_distribution(self, obs, action_masks=None):
        mask = np.asarray(action_masks, dtype=np.int8).reshape(-1)
        legal = np.flatnonzero(mask)
        probs = np.zeros(len(mask), dtype=np.float64)
        probs[legal] = 1.0 / len(legal)
        return _MockDistribution(probs)

    def predict_values(self, obs):
        import torch

        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[np.newaxis, :]
        if self._value_from_hp:
            signal = obs_arr[:, 0]
        else:
            strength_idx = 4 + _POWER_ID_TO_PLAYER_IDX[PowerId.STRENGTH]
            signal = obs_arr[:, strength_idx] if obs_arr.shape[1] > strength_idx else 0.0
        return torch.as_tensor(signal, dtype=torch.float32).reshape(-1, 1)


class MockCombatModel:
    device = "cpu"

    def __init__(self, *, value_from_hp: bool = False):
        self.policy = _MockPolicy(value_from_hp=value_from_hp)

    def predict(self, obs, action_masks=None, deterministic=True):
        dist = self.policy.get_distribution(obs, action_masks=action_masks)
        probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)
        mask = np.asarray(action_masks, dtype=np.int8).reshape(-1)
        legal = np.flatnonzero(mask)
        strike_actions = [a for a in legal if a > 10]
        return int(strike_actions[0] if strike_actions else legal[0]), None


def _sequencing_combat() -> CombatState:
    reset_instance_counter()
    deck = [make_inflame(), make_strike_ironclad()]
    combat = CombatState(player_hp=80, player_max_hp=80, deck=deck, rng_seed=99)
    creature, ai = create_shrinker_beetle(Rng(99))
    combat.add_enemy(creature, ai)
    combat.start_combat()
    combat.hand.clear()
    combat.hand.extend([make_inflame(), make_strike_ironclad()])
    combat.energy = 2
    return combat


def _defend_vs_attack_combat() -> CombatState:
    reset_instance_counter()
    deck = [make_defend_ironclad(), make_strike_ironclad()]
    combat = CombatState(player_hp=12, player_max_hp=80, deck=deck, rng_seed=7)
    creature, ai = create_shrinker_beetle(Rng(7))
    combat.add_enemy(creature, ai)
    combat.start_combat()
    combat.hand.clear()
    combat.hand.extend([make_defend_ironclad(), make_strike_ironclad()])
    combat.energy = 1
    ai.set_forced_move("CHOMP_MOVE")
    return combat


def test_mock_policy_prefers_strike_first():
    combat = _sequencing_combat()
    model = MockCombatModel()
    mask = get_action_mask(combat)
    action, _ = model.predict(
        __import__("sts2_env.gym_env.observation", fromlist=["encode_observation"]).encode_observation(combat),
        action_masks=mask,
        deterministic=True,
    )
    assert action == 16


def test_mcts_prefers_inflame_before_strike():
    combat = _sequencing_combat()
    model = MockCombatModel()
    config = MCTSConfig(n_simulations=64, c_puct=1.5, dirichlet_epsilon=0.0)
    action = mcts_search(combat, model, config)
    assert action == 1


def test_mcts_returns_legal_action(simple_combat):
    model = MockCombatModel()
    config = MCTSConfig(n_simulations=16, dirichlet_epsilon=0.0)
    action = mcts_search(simple_combat, model, config)
    mask = get_action_mask(simple_combat)
    assert mask[action] == 1


def test_mcts_enemy_phase_reduces_damage():
    combat = _defend_vs_attack_combat()
    model = MockCombatModel(value_from_hp=True)
    config = MCTSConfig(
        n_simulations=96, c_puct=1.5, lookahead_player_turns=1, dirichlet_epsilon=0.0,
    )
    action = mcts_search(combat, model, config)
    assert action == 1


def test_mcts_turn2_expansion_reaches_hand():
    combat = _sequencing_combat()
    model = MockCombatModel()
    config = MCTSConfig(n_simulations=8, lookahead_player_turns=1, dirichlet_epsilon=0.0)
    root = _MCTSNode(combat, player_turn_index=0, actions_this_turn=0)
    _expand_all_for_tests(root, model, config)
    end_child = root.children[ACTION_END_TURN]
    assert end_child.player_turn_index == 1
    assert end_child.actions_this_turn == 0
    assert not end_child.terminal
    assert get_action_mask(end_child.combat).any()


def test_search_end_turn_does_not_draw_before_lookahead_resume():
    combat = _sequencing_combat()
    model = MockCombatModel()
    config = MCTSConfig(lookahead_player_turns=1)
    work = clone_combat_state(combat)
    turn_index, actions, terminal, _cached = _apply_search_action(
        work,
        ACTION_END_TURN,
        player_turn_index=0,
        actions_this_turn=0,
        config=config,
        model=model,
    )
    assert turn_index == 1
    assert actions == 0
    assert terminal is False
    assert work.current_side == CombatSide.PLAYER
    assert len(work.hand) > 0

    post_enemy = clone_combat_state(combat)
    post_enemy.finish_player_turn_only()
    assert post_enemy.is_over is False
    post_enemy.advance_enemy_phase(resume_player_turn=False)
    assert len(post_enemy.hand) == 0


def test_apply_root_dirichlet_noise_mixes_priors():
    priors = {0: 0.9, 1: 0.1}
    config = MCTSConfig(dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    np.random.seed(0)
    _apply_root_dirichlet_noise(priors, config)
    assert priors != {0: 0.9, 1: 0.1}
    assert abs(sum(priors.values()) - 1.0) < 1e-9


class _SkewedPolicy:
    def get_distribution(self, obs, action_masks=None):
        mask = np.asarray(action_masks, dtype=np.int8).reshape(-1)
        legal = np.flatnonzero(mask)
        probs = np.zeros(len(mask), dtype=np.float64)
        for action in legal:
            probs[action] = 0.01
        if len(legal) >= 2:
            probs[int(legal[0])] = 0.99
        else:
            probs[legal[0]] = 1.0
        return _MockDistribution(probs)

    def predict_values(self, obs):
        import torch

        return torch.zeros(1, 1, dtype=torch.float32)


class _SkewedModel:
    device = "cpu"

    def __init__(self):
        self.policy = _SkewedPolicy()

    def predict(self, obs, action_masks=None, deterministic=True):
        dist = self.policy.get_distribution(obs, action_masks=action_masks)
        probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)
        return int(np.argmax(probs)), None


def test_mcts_root_dirichlet_increases_low_prior_visits():
    combat = _sequencing_combat()
    model = _SkewedModel()
    np.random.seed(42)
    stats_noisy = MCTSStats()
    mcts_search(
        combat,
        model,
        MCTSConfig(n_simulations=128, dirichlet_epsilon=0.25),
        stats=stats_noisy,
    )
    np.random.seed(42)
    stats_clean = MCTSStats()
    mcts_search(
        combat,
        model,
        MCTSConfig(n_simulations=128, dirichlet_epsilon=0.0),
        stats=stats_clean,
    )
    low_prior_actions = [
        action
        for action, prior in stats_clean.root_priors.items()
        if prior < 0.05
    ]
    assert low_prior_actions
    noisy_low_visits = sum(stats_noisy.root_visits.get(action, 0) for action in low_prior_actions)
    clean_low_visits = sum(stats_clean.root_visits.get(action, 0) for action in low_prior_actions)
    assert noisy_low_visits > clean_low_visits

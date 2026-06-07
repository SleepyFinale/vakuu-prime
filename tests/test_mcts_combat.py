"""MCTS combat sequencing tests."""

from __future__ import annotations

import numpy as np

from sts2_env.cards.base import reset_instance_counter
from sts2_env.cards.ironclad import make_inflame
from sts2_env.cards.ironclad_basic import make_strike_ironclad
from sts2_env.core.combat import CombatState
from sts2_env.core.rng import Rng
from sts2_env.gym_env.action_space import get_action_mask
from sts2_env.monsters.act1_weak import create_shrinker_beetle
from sts2_env.search.mcts_combat import MCTSConfig, mcts_search


class _MockDistribution:
    def __init__(self, probs: np.ndarray):
        import torch

        self.distribution = type("Dist", (), {})()
        self.distribution.probs = torch.as_tensor(probs, dtype=torch.float32).unsqueeze(0)


class _MockPolicy:
    """Policy with uniform priors; ``predict`` still prefers the strike action."""

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
        # Higher player strength index in obs -> higher value.
        strength_idx = 4 + 0  # first tracked power is strength at idx 4
        strength = obs_arr[:, strength_idx] if obs_arr.shape[1] > strength_idx else 0.0
        return torch.as_tensor(strength, dtype=torch.float32).reshape(-1, 1)


class MockCombatModel:
    device = "cpu"

    def __init__(self):
        self.policy = _MockPolicy()

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
    config = MCTSConfig(n_simulations=64, c_puct=1.5)
    action = mcts_search(combat, model, config)
    assert action == 1


def test_mcts_returns_legal_action(simple_combat):
    model = MockCombatModel()
    config = MCTSConfig(n_simulations=16)
    action = mcts_search(simple_combat, model, config)
    mask = get_action_mask(simple_combat)
    assert mask[action] == 1

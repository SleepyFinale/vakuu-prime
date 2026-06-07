"""Tests for Navigator observation encoding (obs v2, 166 dims)."""

import numpy as np
import pytest

from sts2_env.cards.factory import create_card
from sts2_env.cards.ironclad_basic import make_strike_ironclad
from sts2_env.core.enums import CardId, CardRarity, CardType, RelicRarity
from sts2_env.gym_env.navigator_env import STS2NavigatorEnv
from sts2_env.gym_env.navigator_observation import (
    COMBAT_CARD_FEATURES,
    MAX_CARD_OPTIONS,
    MAX_MAP_OPTIONS,
    MAP_OPTION_SIZE,
    NAVIGATOR_OBS_SIZE,
    PATH_TOPOLOGY_SIZE,
    PHASE_OPTION_SIZE,
    RUN_CONTEXT_NAV_SIZE,
    SHOP_FEATURE_SIZE,
    encode_navigator_observation,
)
from sts2_env.gym_env.observation import CARD_FEATURES
from sts2_env.gym_env.run_env import NUM_PHASES
from sts2_env.run.run_manager import RunManager
from sts2_env.run.shop import ShopCardEntry, ShopInventory, ShopRelicEntry

# Slice layout (Navigator obs v2)
_PATH_START = RUN_CONTEXT_NAV_SIZE + NUM_PHASES + MAX_MAP_OPTIONS * MAP_OPTION_SIZE
_CARD_START = _PATH_START + PATH_TOPOLOGY_SIZE
_SHOP_START = _CARD_START + MAX_CARD_OPTIONS * COMBAT_CARD_FEATURES
_PHASE_OPT_START = _SHOP_START + SHOP_FEATURE_SIZE
_DECK_VALUE_IDX = _PHASE_OPT_START + PHASE_OPTION_SIZE


class _MockCombatModel:
    def predict(self, obs, action_masks=None, deterministic=True):
        mask = action_masks if action_masks is not None else np.ones(115, dtype=np.int8)
        valid = np.where(mask == 1)[0]
        return int(valid[0]), None


@pytest.fixture
def mock_nav_env():
    return STS2NavigatorEnv(
        combat_model=_MockCombatModel(),
        reward_shaping=True,
        act_count=3,
        max_steps=500,
    )


def test_obs_size_constant():
    assert NAVIGATOR_OBS_SIZE == 166
    assert COMBAT_CARD_FEATURES == CARD_FEATURES == 9
    assert _DECK_VALUE_IDX == NAVIGATOR_OBS_SIZE - 1


def test_encode_none_returns_zeros():
    obs = encode_navigator_observation(None)
    assert obs.shape == (NAVIGATOR_OBS_SIZE,)
    assert obs.sum() == 0.0


def test_path_topology_on_map(mock_nav_env):
    mock_nav_env.reset(seed=42)
    mgr = mock_nav_env._run_env._mgr
    assert mgr.phase == RunManager.PHASE_MAP_CHOICE

    obs = encode_navigator_observation(mgr)
    path = obs[_PATH_START: _PATH_START + PATH_TOPOLOGY_SIZE]

    assert path[0] > 0.0, "floors_to_boss should be positive at map start"
    assert path[4] == pytest.approx(0.0), "current_act_norm at act 0"
    assert np.all(path >= 0.0)
    assert np.all(path <= 10.0)


def test_card_offers_use_combat_features():
    mgr = RunManager(seed=10, character_id="Ironclad")
    mgr._phase = RunManager.PHASE_CARD_REWARD
    strike = make_strike_ironclad()
    strike.card_type = CardType.ATTACK
    power = create_card(CardId.INFLAME)
    power.card_type = CardType.POWER
    mgr._offered_cards = [strike, power]

    obs = encode_navigator_observation(mgr)
    card_block = obs[_CARD_START: _CARD_START + MAX_CARD_OPTIONS * COMBAT_CARD_FEATURES]

    strike_slot = card_block[:COMBAT_CARD_FEATURES]
    assert strike_slot[0] > 0.0, "card_id norm should be non-zero"
    assert strike_slot[2] > 0.0, "strike should encode damage"
    assert strike_slot[4] == 1.0, "strike is_attack flag"


def test_shop_gold_encoding():
    mgr = RunManager(seed=804, character_id="Ironclad")
    mgr._phase = RunManager.PHASE_SHOP
    mgr.run_state.player.gold = 250
    mgr._shop_inventory = ShopInventory(
        cards=[ShopCardEntry(rarity=CardRarity.COMMON, card_type="Attack", price=50)],
        relics=[ShopRelicEntry(relic_rarity=RelicRarity.STARTER, relic_id="BURNING_BLOOD", price=150)],
        removal_cost=75,
    )

    obs = encode_navigator_observation(mgr)
    shop = obs[_SHOP_START: _SHOP_START + SHOP_FEATURE_SIZE]

    gold_norm = shop[SHOP_FEATURE_SIZE - 2]
    max_price_norm = shop[SHOP_FEATURE_SIZE - 1]

    assert gold_norm == pytest.approx(250 / 1000.0)
    assert max_price_norm == pytest.approx(150 / 1000.0)


def test_navigator_env_obs_shape(mock_nav_env):
    obs, _ = mock_nav_env.reset(seed=42)
    assert obs.shape == (NAVIGATOR_OBS_SIZE,)
    assert mock_nav_env.observation_space.shape == (NAVIGATOR_OBS_SIZE,)

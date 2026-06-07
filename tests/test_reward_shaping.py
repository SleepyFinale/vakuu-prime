"""Tests for non-linear HP shaping and combat micro-rewards."""

import pytest

from sts2_env.core.enums import PowerId, ValueProp
from sts2_env.gym_env.reward import compute_reward
from sts2_env.gym_env.reward_shaping import (
    CombatEventCursor,
    CombatMicroRewardConfig,
    CombatRewardConfig,
    HpShapingConfig,
    compute_combat_hp_step_penalty,
    compute_combat_step_shaping,
    compute_hp_loss_penalty,
    hp_marginal_weight,
)
from sts2_env.gym_env.run_reward import RunRewardConfig, RunRewardSnapshot, compute_run_shaping
from sts2_env.run.run_manager import RunManager


class TestHpMarginalWeight:
    def test_weight_increases_as_hp_drops(self):
        assert hp_marginal_weight(1.0, 3.0) < hp_marginal_weight(0.125, 3.0)

    def test_same_damage_penalized_more_at_low_hp(self):
        config = HpShapingConfig(penalty_scale=0.2, max_penalty=1.0, steepness=3.0)
        hp_lost = 5
        max_hp = 80
        high_hp_penalty = compute_hp_loss_penalty(hp_lost, max_hp, 1.0, config)
        low_hp_penalty = compute_hp_loss_penalty(hp_lost, max_hp, 0.125, config)
        assert low_hp_penalty > high_hp_penalty

    def test_small_hit_at_full_hp_is_minor(self):
        config = HpShapingConfig()
        penalty = compute_hp_loss_penalty(5, 80, 1.0, config)
        assert penalty == pytest.approx(0.0125, abs=0.002)

    def test_respects_max_penalty(self):
        config = HpShapingConfig(penalty_scale=1.0, max_penalty=0.05, steepness=5.0)
        penalty = compute_hp_loss_penalty(40, 80, 0.1, config)
        assert penalty == pytest.approx(0.05)


class TestCombatMicroRewards:
    def test_vulnerable_applied_by_player_to_enemy(self, simple_combat):
        combat = simple_combat
        player = combat.primary_player
        enemy = combat.enemies[0]
        config = CombatMicroRewardConfig(vulnerable_scale=0.02)

        combat.apply_power_to(enemy, PowerId.VULNERABLE, 2, applier=player)
        reward, _ = compute_combat_step_shaping(
            combat, CombatEventCursor(), config,
        )
        assert reward == pytest.approx(0.04)

    def test_weak_applied_by_player_to_enemy(self, simple_combat):
        combat = simple_combat
        player = combat.primary_player
        enemy = combat.enemies[0]
        config = CombatMicroRewardConfig(weak_scale=0.03)

        combat.apply_power_to(enemy, PowerId.WEAK, 1, applier=player)
        reward, _ = compute_combat_step_shaping(
            combat, CombatEventCursor(), config,
        )
        assert reward == pytest.approx(0.03)

    def test_no_reward_for_enemy_debuff_on_player(self, simple_combat):
        combat = simple_combat
        player = combat.primary_player
        enemy = combat.enemies[0]
        config = CombatMicroRewardConfig()

        combat.apply_power_to(player, PowerId.WEAK, 2, applier=enemy)
        reward, _ = compute_combat_step_shaping(
            combat, CombatEventCursor(), config,
        )
        assert reward == 0.0

    def test_block_absorbed_from_enemy_attack(self, simple_combat):
        combat = simple_combat
        player = combat.primary_player
        enemy = combat.enemies[0]
        config = CombatMicroRewardConfig(block_scale=0.001)

        player.gain_block(10)
        combat.deal_damage(enemy, player, 8, ValueProp.MOVE)

        reward, _ = compute_combat_step_shaping(
            combat, CombatEventCursor(), config,
        )
        assert reward == pytest.approx(0.008)

    def test_micro_reward_capped_per_step(self, simple_combat):
        combat = simple_combat
        player = combat.primary_player
        enemy = combat.enemies[0]
        config = CombatMicroRewardConfig(
            vulnerable_scale=0.02,
            max_step_bonus=0.05,
        )

        combat.apply_power_to(enemy, PowerId.VULNERABLE, 10, applier=player)
        reward, _ = compute_combat_step_shaping(
            combat, CombatEventCursor(), config,
        )
        assert reward == pytest.approx(0.05)

    def test_cursor_only_counts_new_events(self, simple_combat):
        combat = simple_combat
        player = combat.primary_player
        enemy = combat.enemies[0]
        config = CombatMicroRewardConfig(vulnerable_scale=0.02)

        combat.apply_power_to(enemy, PowerId.VULNERABLE, 1, applier=player)
        _, cursor = compute_combat_step_shaping(
            combat, CombatEventCursor(), config,
        )

        combat.apply_power_to(enemy, PowerId.VULNERABLE, 1, applier=player)
        reward, _ = compute_combat_step_shaping(combat, cursor, config)
        assert reward == pytest.approx(0.02)


class TestCombatEnvRewardIntegration:
    def test_shaping_disabled_is_sparse(self, simple_combat):
        combat = simple_combat
        reward, _ = compute_reward(
            combat, combat.player.current_hp, reward_shaping=False,
        )
        assert reward == 0.0

    def test_hp_step_penalty_on_damage(self, simple_combat):
        combat = simple_combat
        player = combat.primary_player
        prev_hp = player.current_hp
        player.lose_hp(5, fire_hooks=False)

        penalty = compute_combat_hp_step_penalty(prev_hp, combat, HpShapingConfig())
        assert penalty > 0.0

        reward, _ = compute_reward(
            combat,
            prev_hp,
            reward_shaping=True,
            reward_config=CombatRewardConfig(),
        )
        assert reward < 0.0


class TestRunRewardNonlinearHp:
    def test_combat_clear_uses_nonlinear_hp_penalty(self):
        config = RunRewardConfig(
            combat_clear_bonus=0.1,
            hp=HpShapingConfig(penalty_scale=0.2, max_penalty=0.2, steepness=3.0),
        )
        max_hp = 100
        prev = RunRewardSnapshot(
            total_floor=4,
            hp_ratio=0.8,
            max_hp=max_hp,
            phase=RunManager.PHASE_COMBAT,
            combat_active=True,
        )
        curr = RunRewardSnapshot(
            total_floor=4,
            hp_ratio=0.5,
            max_hp=max_hp,
            phase=RunManager.PHASE_CARD_REWARD,
            combat_active=False,
            last_combat_won=True,
        )
        hp_lost = 30
        expected_penalty = compute_hp_loss_penalty(
            hp_lost, max_hp, prev.hp_ratio, config.hp,
        )
        reward = compute_run_shaping(prev, curr, config)
        assert reward == pytest.approx(0.1 - expected_penalty)
        assert expected_penalty > 0.06

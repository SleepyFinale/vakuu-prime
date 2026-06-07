"""Tests for non-linear HP shaping and combat micro-rewards."""

import pytest

from sts2_env.core.enums import PowerId, ValueProp
from sts2_env.core.rng import Rng
from sts2_env.gym_env.reward import compute_reward
from sts2_env.gym_env.reward_shaping import (
    CombatEventCursor,
    CombatMicroRewardConfig,
    CombatRewardConfig,
    HpShapingConfig,
    compute_combat_flawless_bonus,
    compute_combat_hp_step_penalty,
    compute_combat_kill_reward,
    compute_combat_step_shaping,
    compute_hp_loss_penalty,
    debuff_stack_marginal,
    hp_marginal_weight,
)
from sts2_env.monsters.act1_weak import create_shrinker_beetle
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
        expected = debuff_stack_marginal(0, 2) * 0.02
        assert reward == pytest.approx(expected)

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
        expected = debuff_stack_marginal(1, 1) * 0.02
        assert reward == pytest.approx(expected)

    def test_debuff_marginal_diminishes_at_high_stacks(self, simple_combat):
        combat = simple_combat
        player = combat.primary_player
        enemy = combat.enemies[0]
        config = CombatMicroRewardConfig(vulnerable_scale=0.02)

        combat.apply_power_to(enemy, PowerId.VULNERABLE, 10, applier=player)
        _, cursor = compute_combat_step_shaping(
            combat, CombatEventCursor(), config,
        )

        combat.apply_power_to(enemy, PowerId.VULNERABLE, 3, applier=player)
        reward, _ = compute_combat_step_shaping(combat, cursor, config)
        expected = debuff_stack_marginal(10, 3) * 0.02
        linear = 3 * 0.02
        assert reward == pytest.approx(expected)
        assert reward < linear

    def test_debuff_marginal_multiple_events_same_step(self, simple_combat):
        combat = simple_combat
        player = combat.primary_player
        enemy = combat.enemies[0]
        config = CombatMicroRewardConfig(vulnerable_scale=0.02)

        combat.apply_power_to(enemy, PowerId.VULNERABLE, 2, applier=player)
        combat.apply_power_to(enemy, PowerId.VULNERABLE, 1, applier=player)
        reward, _ = compute_combat_step_shaping(
            combat, CombatEventCursor(), config,
        )
        expected = (
            debuff_stack_marginal(0, 2) + debuff_stack_marginal(2, 1)
        ) * 0.02
        assert reward == pytest.approx(expected)

    def test_debuff_stack_marginal_edge_cases(self):
        assert debuff_stack_marginal(0, 0) == 0.0
        assert debuff_stack_marginal(5, 0) == 0.0
        assert debuff_stack_marginal(0, 1) == pytest.approx(1.0)


class TestCombatKillReward:
    def test_single_kill_mid_fight(self, simple_combat):
        combat = simple_combat
        enemy = combat.enemies[0]
        config = CombatMicroRewardConfig(kill_scale=0.05)

        combat.kill_creature(enemy)
        assert not combat.is_over or len(combat.alive_enemies) == 0

        reward = compute_combat_kill_reward(1, combat, config)
        assert reward == pytest.approx(0.05)

    def test_no_deaths(self, simple_combat):
        combat = simple_combat
        config = CombatMicroRewardConfig(kill_scale=0.05)

        reward = compute_combat_kill_reward(len(combat.alive_enemies), combat, config)
        assert reward == 0.0

    def test_multi_kill_same_step(self, simple_combat, rng):
        combat = simple_combat
        for offset in (100, 200):
            creature, ai = create_shrinker_beetle(Rng(offset))
            combat.add_enemy(creature, ai)
        config = CombatMicroRewardConfig(kill_scale=0.05)
        prev_alive = len(combat.alive_enemies)
        assert prev_alive == 3

        for enemy in list(combat.alive_enemies):
            combat.kill_creature(enemy)

        reward = compute_combat_kill_reward(prev_alive, combat, config)
        assert reward == pytest.approx(0.15)

    def test_revive_does_not_penalize(self, simple_combat, rng):
        combat = simple_combat
        creature, ai = create_shrinker_beetle(Rng(100))
        combat.add_enemy(creature, ai)
        config = CombatMicroRewardConfig(kill_scale=0.05)

        reward = compute_combat_kill_reward(1, combat, config)
        assert reward == 0.0

    def test_terminal_win_includes_kill_shaping(self, simple_combat):
        combat = simple_combat
        enemy = combat.enemies[0]
        prev_hp = combat.player.current_hp
        prev_alive = len(combat.alive_enemies)

        combat.kill_creature(enemy)
        assert combat.is_over
        assert combat.player_won

        reward, _ = compute_reward(
            combat,
            prev_hp,
            reward_shaping=True,
            reward_config=CombatRewardConfig(),
            prev_alive_count=prev_alive,
            combat_gross_hp_lost=0,
        )
        assert reward == pytest.approx(1.15)

    def test_terminal_win_sparse_when_shaping_disabled(self, simple_combat):
        combat = simple_combat
        enemy = combat.enemies[0]
        prev_hp = combat.player.current_hp
        prev_alive = len(combat.alive_enemies)

        combat.kill_creature(enemy)

        reward, _ = compute_reward(
            combat,
            prev_hp,
            reward_shaping=False,
            prev_alive_count=prev_alive,
        )
        assert reward == pytest.approx(1.0)


class TestCombatFlawlessBonus:
    def test_helper_awards_on_flawless_win(self):
        assert compute_combat_flawless_bonus(
            player_won=True, gross_hp_lost=0, bonus=0.1,
        ) == pytest.approx(0.1)

    def test_helper_skips_after_damage(self):
        assert compute_combat_flawless_bonus(
            player_won=True, gross_hp_lost=5, bonus=0.1,
        ) == 0.0

    def test_flawless_win_includes_bonus(self, simple_combat):
        combat = simple_combat
        enemy = combat.enemies[0]
        prev_hp = combat.player.current_hp
        prev_alive = len(combat.alive_enemies)

        combat.kill_creature(enemy)

        reward, _ = compute_reward(
            combat,
            prev_hp,
            reward_shaping=True,
            reward_config=CombatRewardConfig(flawless_bonus=0.1),
            prev_alive_count=prev_alive,
            combat_gross_hp_lost=0,
        )
        assert reward == pytest.approx(1.15)

    def test_win_after_damage_has_no_flawless_bonus(self, simple_combat):
        combat = simple_combat
        player = combat.primary_player
        player.lose_hp(5, fire_hooks=False)
        enemy = combat.enemies[0]
        prev_hp = player.current_hp
        prev_alive = len(combat.alive_enemies)

        combat.kill_creature(enemy)

        reward, _ = compute_reward(
            combat,
            prev_hp,
            reward_shaping=True,
            reward_config=CombatRewardConfig(flawless_bonus=0.1),
            prev_alive_count=prev_alive,
            combat_gross_hp_lost=5,
        )
        assert reward == pytest.approx(1.05)

    def test_damage_then_heal_has_no_flawless_bonus(self, simple_combat):
        combat = simple_combat
        player = combat.primary_player
        start_hp = player.current_hp
        player.lose_hp(10, fire_hooks=False)
        player.heal(10)
        assert player.current_hp == start_hp

        enemy = combat.enemies[0]
        prev_hp = player.current_hp
        prev_alive = len(combat.alive_enemies)
        combat.kill_creature(enemy)

        reward, _ = compute_reward(
            combat,
            prev_hp,
            reward_shaping=True,
            reward_config=CombatRewardConfig(flawless_bonus=0.1),
            prev_alive_count=prev_alive,
            combat_gross_hp_lost=10,
        )
        assert reward == pytest.approx(1.05)

    def test_hard_start_flawless_at_partial_hp(self, rng):
        from sts2_env.cards.ironclad_basic import create_ironclad_starter_deck
        from sts2_env.core.combat import CombatState
        from sts2_env.monsters.act1_weak import create_shrinker_beetle

        deck = create_ironclad_starter_deck()
        combat = CombatState(player_hp=20, player_max_hp=80, deck=deck, rng_seed=42)
        creature, ai = create_shrinker_beetle(rng)
        combat.add_enemy(creature, ai)
        combat.start_combat()

        enemy = combat.enemies[0]
        prev_hp = combat.player.current_hp
        prev_alive = len(combat.alive_enemies)
        combat.kill_creature(enemy)

        reward, _ = compute_reward(
            combat,
            prev_hp,
            reward_shaping=True,
            reward_config=CombatRewardConfig(flawless_bonus=0.1),
            prev_alive_count=prev_alive,
            combat_gross_hp_lost=0,
        )
        assert reward == pytest.approx(1.15)

    def test_shaping_disabled_skips_flawless_bonus(self, simple_combat):
        combat = simple_combat
        enemy = combat.enemies[0]
        prev_hp = combat.player.current_hp
        prev_alive = len(combat.alive_enemies)

        combat.kill_creature(enemy)

        reward, _ = compute_reward(
            combat,
            prev_hp,
            reward_shaping=False,
            prev_alive_count=prev_alive,
            combat_gross_hp_lost=0,
        )
        assert reward == pytest.approx(1.0)


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
        reward = compute_run_shaping(prev, curr, config, combat_gross_hp_lost=30)
        assert reward == pytest.approx(0.1 - expected_penalty)
        assert expected_penalty > 0.06

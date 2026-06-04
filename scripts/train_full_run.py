"""Train a MaskablePPO meta-policy on the STS2 hierarchical full-run environment.

Combat is delegated to a pre-trained combat MaskablePPO; this script trains
map navigation, card rewards, shops, and other non-combat decisions.

Usage:
    pip install "sts2-rl-agent[train]"
    python scripts/train_combat.py --output-dir output/combat_ppo
    python scripts/train_full_run.py \\
        --combat-model output/combat_ppo/best_model/best_model.zip

Requires: stable-baselines3, sb3-contrib, torch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_COMBAT_MODEL = "output/combat_ppo/best_model/best_model.zip"
DEFAULT_MAX_STEPS = 10_000


def make_env(
    seed: int = 0,
    act_count: int = 1,
    reward_shaping: bool = True,
    combat_model: str | None = None,
    delegate_combat: bool = True,
    max_steps: int = DEFAULT_MAX_STEPS,
):
    """Create a single STS2HierarchicalRunEnv."""
    from sts2_env.gym_env.hierarchical_run_env import STS2HierarchicalRunEnv

    def _init():
        env = STS2HierarchicalRunEnv(
            combat_model_path=combat_model,
            delegate_combat=delegate_combat,
            act_count=act_count,
            reward_shaping=reward_shaping,
            max_steps=max_steps,
        )
        env.reset(seed=seed)
        return env

    return _init


def make_masked_env(
    seed: int,
    act_count: int = 1,
    reward_shaping: bool = True,
    combat_model: str | None = None,
    delegate_combat: bool = True,
    max_steps: int = DEFAULT_MAX_STEPS,
):
    """Create a masked env factory for vectorised envs."""
    try:
        from sb3_contrib.common.wrappers import ActionMasker
    except ImportError:
        print("Training requires sb3-contrib and stable-baselines3.")
        print("Install with: pip install 'sts2-rl-agent[train]'")
        sys.exit(1)

    def mask_fn(env):
        return env.action_masks()

    def _init():
        if delegate_combat:
            from sts2_env.gym_env.hierarchical_run_env import STS2HierarchicalRunEnv

            env = STS2HierarchicalRunEnv(
                combat_model_path=combat_model,
                delegate_combat=True,
                act_count=act_count,
                reward_shaping=reward_shaping,
                max_steps=max_steps,
            )
        else:
            from sts2_env.gym_env.run_env import STS2RunEnv

            env = STS2RunEnv(
                act_count=act_count,
                reward_shaping=reward_shaping,
                max_steps=max_steps,
            )
        env = ActionMasker(env, mask_fn)
        return env

    return _init


def train(args):
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("Training requires sb3-contrib and stable-baselines3.")
        print("Install with: pip install 'sts2-rl-agent[train]'")
        sys.exit(1)

    combat_model = args.combat_model
    if args.delegate_combat:
        if combat_model is None:
            combat_model = DEFAULT_COMBAT_MODEL
        if not Path(combat_model).exists():
            print(f"Combat model not found: {combat_model}")
            print("Train combat first: python scripts/train_combat.py")
            sys.exit(1)

    print("Training MaskablePPO on STS2 Hierarchical Full Run")
    print(f"  act_count:        {args.act_count}")
    print(f"  n_envs:           {args.n_envs}")
    print(f"  total_timesteps:  {args.total_timesteps}")
    print(f"  learning_rate:    {args.lr}")
    print(f"  batch_size:       {args.batch_size}")
    print(f"  reward_shaping:   {args.reward_shaping}")
    print(f"  delegate_combat:  {args.delegate_combat}")
    print(f"  combat_model:     {combat_model}")
    print(f"  max_steps:        {args.max_steps}")
    print(f"  output_dir:       {args.output_dir}")
    if args.load_model:
        print(f"  load_model:       {args.load_model}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        act_count=args.act_count,
        reward_shaping=args.reward_shaping,
        combat_model=combat_model,
        delegate_combat=args.delegate_combat,
        max_steps=args.max_steps,
    )

    if args.n_envs > 1:
        train_env = SubprocVecEnv([
            make_masked_env(i, **env_kwargs)
            for i in range(args.n_envs)
        ])
    else:
        train_env = DummyVecEnv([
            make_masked_env(0, **env_kwargs)
        ])

    eval_env = DummyVecEnv([
        make_masked_env(9999, **env_kwargs)
    ])

    if args.load_model:
        model = MaskablePPO.load(
            args.load_model,
            env=train_env,
            tensorboard_log=str(output_dir / "tb_logs"),
        )
        reset_timesteps = False
    else:
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=str(output_dir / "tb_logs"),
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
        )
        reset_timesteps = True

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=False,
    )

    start = time.perf_counter()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=True,
        reset_num_timesteps=reset_timesteps,
    )
    elapsed = time.perf_counter() - start

    final_path = str(output_dir / "final_model")
    model.save(final_path)
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {output_dir / 'best_model'}")

    print("\n--- Final Evaluation (sparse rewards) ---")
    evaluate(
        model,
        act_count=args.act_count,
        combat_model=combat_model,
        delegate_combat=args.delegate_combat,
        n_episodes=100,
    )

    train_env.close()
    eval_env.close()


def _make_eval_env(
    act_count: int,
    combat_model: str | None,
    delegate_combat: bool,
):
    from sb3_contrib.common.wrappers import ActionMasker

    if delegate_combat:
        from sts2_env.gym_env.hierarchical_run_env import STS2HierarchicalRunEnv

        base = STS2HierarchicalRunEnv(
            combat_model_path=combat_model or DEFAULT_COMBAT_MODEL,
            delegate_combat=True,
            act_count=act_count,
            reward_shaping=False,
            max_steps=DEFAULT_MAX_STEPS,
        )
    else:
        from sts2_env.gym_env.run_env import STS2RunEnv

        base = STS2RunEnv(
            act_count=act_count,
            reward_shaping=False,
            max_steps=DEFAULT_MAX_STEPS,
        )

    def mask_fn(env):
        return env.action_masks()

    return ActionMasker(base, mask_fn)


def evaluate(
    model,
    act_count: int = 1,
    combat_model: str | None = None,
    delegate_combat: bool = True,
    n_episodes: int = 100,
):
    """Evaluate with sparse terminal rewards only."""
    env = _make_eval_env(act_count, combat_model, delegate_combat)

    wins = 0
    total_rewards = []
    floors_reached = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 10000)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            steps += 1
            done = terminated or truncated
            if terminated and reward > 0:
                wins += 1
        total_rewards.append(ep_reward)
        episode_lengths.append(steps)
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "run_state") and unwrapped.run_state is not None:
            floors_reached.append(unwrapped.run_state.total_floor)
        else:
            floors_reached.append(0)

    print(f"Episodes:         {n_episodes}")
    print(f"Win rate:         {wins / n_episodes:.1%}")
    print(f"Avg reward:       {np.mean(total_rewards):.3f}")
    print(f"Avg ep length:    {np.mean(episode_lengths):.1f}")
    print(f"Avg floors:       {np.mean(floors_reached):.1f}")
    print(f"Max floors:       {max(floors_reached)}")


def random_baseline(
    act_count: int = 1,
    combat_model: str | None = None,
    delegate_combat: bool = True,
    n_episodes: int = 100,
):
    """Random-action baseline with sparse rewards."""
    if delegate_combat:
        from sts2_env.gym_env.hierarchical_run_env import STS2HierarchicalRunEnv

        env = STS2HierarchicalRunEnv(
            combat_model_path=combat_model or DEFAULT_COMBAT_MODEL,
            delegate_combat=True,
            act_count=act_count,
            reward_shaping=False,
            max_steps=DEFAULT_MAX_STEPS,
        )
    else:
        from sts2_env.gym_env.run_env import STS2RunEnv

        env = STS2RunEnv(
            act_count=act_count,
            reward_shaping=False,
            max_steps=DEFAULT_MAX_STEPS,
        )
    rng = np.random.RandomState(42)

    wins = 0
    total_rewards = []
    floors_reached = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        while not done:
            mask = info["action_mask"]
            valid = np.where(mask == 1)[0]
            action = int(rng.choice(valid))
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            if terminated and reward > 0:
                wins += 1
        total_rewards.append(ep_reward)
        floors_reached.append(env.run_state.total_floor if env.run_state else 0)

    print("=== Random Baseline (hierarchical, sparse eval) ===")
    print(f"Episodes:         {n_episodes}")
    print(f"Win rate:         {wins / n_episodes:.1%}")
    print(f"Avg reward:       {np.mean(total_rewards):.3f}")
    print(f"Avg floors:       {np.mean(floors_reached):.1f}")
    print(f"Max floors:       {max(floors_reached)}")


def main():
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO meta-policy on STS2 hierarchical full run",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=1_000_000,
        help="Total training timesteps (default: 1000000)",
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--act-count", type=int, default=1,
        help="Acts per run before curriculum win (1=Act1, 3=full game)",
    )
    parser.add_argument(
        "--combat-model", type=str, default=None,
        help=f"Path to combat MaskablePPO zip (default: {DEFAULT_COMBAT_MODEL})",
    )
    parser.add_argument(
        "--no-combat-delegate", action="store_true",
        help="Train flat policy without combat delegation (ablation)",
    )
    parser.add_argument(
        "--load-model", type=str, default=None,
        help="Resume or fine-tune from a saved meta-policy zip",
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS,
        help="Max meta-steps per episode (default: 10000)",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Minibatch size (default: 256)",
    )
    parser.add_argument(
        "--n-steps", type=int, default=2048,
        help="Steps per rollout per env (default: 2048)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=10,
        help="PPO epochs per update (default: 10)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.995,
        help="Discount factor (default: 0.995)",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.02,
        help="Entropy coefficient (default: 0.02)",
    )
    parser.add_argument(
        "--reward-shaping", action="store_true", default=True,
        help="Use reward shaping (default: True)",
    )
    parser.add_argument(
        "--no-reward-shaping", action="store_false", dest="reward_shaping",
        help="Disable reward shaping (sparse only)",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=20_000,
        help="Evaluate every N steps (default: 20000)",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10,
        help="Episodes per evaluation (default: 10)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/run_ppo",
        help="Output directory (default: output/run_ppo)",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Only run random baseline evaluation (no training)",
    )
    args = parser.parse_args()
    args.delegate_combat = not args.no_combat_delegate

    if args.baseline_only:
        random_baseline(
            act_count=args.act_count,
            combat_model=args.combat_model,
            delegate_combat=args.delegate_combat,
        )
    else:
        print("Running random baseline for reference...")
        random_baseline(
            act_count=args.act_count,
            combat_model=args.combat_model,
            delegate_combat=args.delegate_combat,
            n_episodes=50,
        )
        print()
        train(args)


if __name__ == "__main__":
    main()

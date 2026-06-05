"""Train a MaskablePPO agent on STS2 combat.

Usage:
    pip install "sts2-rl-agent[train]"
    python scripts/train_combat.py

Requires: stable-baselines3, sb3-contrib, torch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_ACTS = "0"
DEFAULT_CHARACTER = "Ironclad"
MIXED_ACTS_DEFAULT_TIMESTEPS = 3_000_000
MIXED_CHARS_DEFAULT_TIMESTEPS = 4_000_000
SINGLE_ACT_DEFAULT_TIMESTEPS = 2_000_000


def parse_acts(acts_spec: str) -> tuple[int, ...]:
    from sts2_env.encounters.pools import parse_act_indices

    return parse_act_indices(acts_spec)


def parse_characters(characters_spec: str) -> tuple[str, ...]:
    from sts2_env.characters.all import parse_character_ids

    return parse_character_ids(characters_spec)


def resolve_training_characters(args) -> tuple[str | None, tuple[str, ...] | None]:
    """Return (fixed character_id, character_ids pool) from CLI args."""
    if args.characters is not None:
        return None, parse_characters(args.characters)
    return args.character, None


def default_output_dir(
    encounter_acts: tuple[int, ...],
    character_id: str | None,
    character_ids: tuple[str, ...] | None,
) -> str:
    if character_ids is not None:
        if len(character_ids) == len(parse_characters("all")):
            return "output/combat_ppo_mixed_chars"
        chars_slug = "_".join(c.lower() for c in character_ids)
        return f"output/combat_ppo_{chars_slug}"

    mixed_acts = len(encounter_acts) > 1 or any(a > 0 for a in encounter_acts)
    if mixed_acts:
        return "output/combat_ppo_mixed"

    if character_id and character_id != DEFAULT_CHARACTER:
        return f"output/combat_ppo_{character_id.lower()}"

    return "output/combat_ppo"


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

    encounter_acts = parse_acts(args.acts)
    character_id, character_ids = resolve_training_characters(args)

    print("Training MaskablePPO on STS2 combat")
    print(f"  acts:            {encounter_acts}")
    print(f"  act1_biome:      {args.act1_biome}")
    if character_ids is not None:
        print(f"  characters:      {character_ids}")
    else:
        print(f"  character:       {character_id}")
    print(f"  n_envs:          {args.n_envs}")
    print(f"  total_timesteps: {args.total_timesteps}")
    print(f"  learning_rate:   {args.lr}")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  output_dir:      {args.output_dir}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def mask_fn(env):
        return env.action_masks()

    def make_masked_env(seed: int):
        def _init():
            from sts2_env.gym_env.combat_env import STS2CombatEnv

            env = STS2CombatEnv(
                encounter_acts=encounter_acts,
                act1_biome=args.act1_biome,
                character_id=character_id or DEFAULT_CHARACTER,
                character_ids=character_ids,
            )
            env = ActionMasker(env, mask_fn)
            return env
        return _init

    if args.n_envs > 1:
        train_env = SubprocVecEnv([
            make_masked_env(i) for i in range(args.n_envs)
        ])
    else:
        train_env = DummyVecEnv([make_masked_env(0)])

    eval_env = DummyVecEnv([make_masked_env(9999)])

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
    )

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
    )
    elapsed = time.perf_counter() - start

    final_path = str(output_dir / "final_model")
    model.save(final_path)
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {output_dir / 'best_model'}")

    print("\n--- Final Evaluation ---")
    evaluate(
        model,
        encounter_acts=encounter_acts,
        act1_biome=args.act1_biome,
        character_id=character_id,
        character_ids=character_ids,
        n_episodes=100,
    )

    train_env.close()
    eval_env.close()


def evaluate(
    model,
    encounter_acts: tuple[int, ...] = (0,),
    act1_biome: str = "random",
    character_id: str | None = DEFAULT_CHARACTER,
    character_ids: tuple[str, ...] | None = None,
    n_episodes: int = 100,
):
    """Evaluate trained model."""
    from sb3_contrib.common.wrappers import ActionMasker
    from sts2_env.gym_env.combat_env import STS2CombatEnv

    def mask_fn(env):
        return env.action_masks()

    env = ActionMasker(
        STS2CombatEnv(
            encounter_acts=encounter_acts,
            act1_biome=act1_biome,
            character_id=character_id or DEFAULT_CHARACTER,
            character_ids=character_ids,
        ),
        mask_fn,
    )
    wins = 0
    total_rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 10000)
        done = False
        ep_reward = 0.0
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated
            if terminated and reward > 0:
                wins += 1
        total_rewards.append(ep_reward)

    print(f"Episodes:    {n_episodes}")
    print(f"Acts:        {encounter_acts}")
    if character_ids is not None:
        print(f"Characters:  {character_ids}")
    else:
        print(f"Character:   {character_id}")
    print(f"Win rate:    {wins / n_episodes:.1%}")
    print(f"Avg reward:  {np.mean(total_rewards):.3f}")


def main():
    from sts2_env.characters.all import SUPPORTED_TRAINING_CHARACTERS

    parser = argparse.ArgumentParser(description="Train MaskablePPO on STS2 combat")
    parser.add_argument(
        "--acts", type=str, default=DEFAULT_ACTS,
        help="Act indices for encounter pool: '0', '0,1,2', or 'all' (default: 0)",
    )
    parser.add_argument(
        "--act1-biome", type=str, default="random",
        choices=("random", "overgrowth", "underdocks"),
        help="Act 1 biome for index 0: random mixes both, or force one (default: random)",
    )
    character_group = parser.add_mutually_exclusive_group()
    character_group.add_argument(
        "--character", type=str, default=DEFAULT_CHARACTER,
        choices=SUPPORTED_TRAINING_CHARACTERS,
        help=f"Single character to train (default: {DEFAULT_CHARACTER})",
    )
    character_group.add_argument(
        "--characters", type=str, default=None,
        help="Mixed-character pool: 'Ironclad,Silent' or 'all'",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=None,
        help="Total training timesteps (default: 2M single, 3M mixed acts, 4M mixed chars)",
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel environments (default: 4)",
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
        "--gamma", type=float, default=0.99,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01,
        help="Entropy coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=10_000,
        help="Evaluate every N steps (default: 10000)",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=20,
        help="Episodes per evaluation (default: 20)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (auto-selected from character/acts if omitted)",
    )
    args = parser.parse_args()

    encounter_acts = parse_acts(args.acts)
    character_id, character_ids = resolve_training_characters(args)

    if args.total_timesteps is None:
        if character_ids is not None:
            args.total_timesteps = MIXED_CHARS_DEFAULT_TIMESTEPS
        elif len(encounter_acts) > 1 or any(a > 0 for a in encounter_acts):
            args.total_timesteps = MIXED_ACTS_DEFAULT_TIMESTEPS
        else:
            args.total_timesteps = SINGLE_ACT_DEFAULT_TIMESTEPS

    if args.output_dir is None:
        args.output_dir = default_output_dir(
            encounter_acts, character_id, character_ids,
        )

    train(args)


if __name__ == "__main__":
    main()

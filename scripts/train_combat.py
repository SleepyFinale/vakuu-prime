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
    *,
    policy: str = "attention",
) -> str:
    if policy == "attention":
        policy_suffix = "_attn"
    elif policy == "gnn":
        policy_suffix = "_gnn"
    else:
        policy_suffix = ""

    if character_ids is not None:
        if len(character_ids) == len(parse_characters("all")):
            return f"output/combat_ppo_mixed_chars{policy_suffix}"
        chars_slug = "_".join(c.lower() for c in character_ids)
        return f"output/combat_ppo_{chars_slug}{policy_suffix}"

    mixed_acts = len(encounter_acts) > 1 or any(a > 0 for a in encounter_acts)
    if mixed_acts:
        return f"output/combat_ppo_mixed{policy_suffix}"

    if character_id and character_id != DEFAULT_CHARACTER:
        return f"output/combat_ppo_{character_id.lower()}{policy_suffix}"

    return f"output/combat_ppo{policy_suffix}"


def build_policy_kwargs(args) -> dict:
    """Build MaskablePPO policy_kwargs from CLI args."""
    if args.policy == "attention":
        from sts2_env.training.attention_extractor import CombatAttentionExtractor

        extractor_class = CombatAttentionExtractor
    elif args.policy == "gnn":
        from sts2_env.training.gnn_extractor import CombatGNNExtractor

        extractor_class = CombatGNNExtractor
    else:
        return {}

    return dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=dict(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            features_dim=args.features_dim,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )


def build_combat_reward_config(args):
    from sts2_env.gym_env.reward_shaping import (
        CombatMicroRewardConfig,
        CombatRewardConfig,
        HpShapingConfig,
    )

    return CombatRewardConfig(
        hp=HpShapingConfig(steepness=args.hp_steepness),
        micro=CombatMicroRewardConfig(
            vulnerable_scale=args.vulnerable_scale,
            weak_scale=args.weak_scale,
            block_scale=args.block_scale,
        ),
    )


def train(args):
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    except ImportError:
        print("Training requires sb3-contrib and stable-baselines3.")
        print("Install with: pip install 'sts2-rl-agent[train]'")
        sys.exit(1)

    from sts2_env.training.checkpointing import (
        build_ppo_callbacks,
        handle_training_keyboard_interrupt,
        print_pause_message,
        print_resume_progress,
        safe_close_vec_envs,
        save_run_config,
    )

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
    print(f"  policy:          {args.policy}")
    if args.policy in ("attention", "gnn"):
        print(f"  d_model:         {args.d_model}")
        print(f"  n_heads:         {args.n_heads}")
        print(f"  n_layers:        {args.n_layers}")
        print(f"  features_dim:    {args.features_dim}")
    print(f"  output_dir:      {args.output_dir}")
    print(f"  reward_shaping:  {args.reward_shaping}")
    if args.reward_shaping:
        print(f"  hp_steepness:    {args.hp_steepness}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not getattr(args, "resume", False):
        save_run_config(output_dir, args)

    def mask_fn(env):
        return env.action_masks()

    reward_config = build_combat_reward_config(args) if args.reward_shaping else None

    def make_masked_env(seed: int, *, shaping: bool | None = None):
        use_shaping = args.reward_shaping if shaping is None else shaping

        def _init():
            from sts2_env.gym_env.combat_env import STS2CombatEnv

            env = STS2CombatEnv(
                encounter_acts=encounter_acts,
                act1_biome=args.act1_biome,
                character_id=character_id or DEFAULT_CHARACTER,
                character_ids=character_ids,
                reward_shaping=use_shaping,
                reward_config=reward_config if use_shaping else None,
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

    eval_env = DummyVecEnv([make_masked_env(9999, shaping=False)])

    if args.load_model:
        model = MaskablePPO.load(
            args.load_model,
            env=train_env,
            tensorboard_log=str(output_dir / "tb_logs"),
        )
        reset_timesteps = False
        print_resume_progress(model, args.total_timesteps)
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
            policy_kwargs=build_policy_kwargs(args),
            verbose=1,
            tensorboard_log=str(output_dir / "tb_logs"),
        )
        reset_timesteps = True

    callbacks, interrupt_callback = build_ppo_callbacks(
        output_dir=output_dir,
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        n_envs=args.n_envs,
        checkpoint_freq=args.checkpoint_freq,
        keep_checkpoints=args.keep_checkpoints,
    )

    start = time.perf_counter()
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=reset_timesteps,
        )
    except KeyboardInterrupt:
        handle_training_keyboard_interrupt(model, interrupt_callback)
    elapsed = time.perf_counter() - start

    if interrupt_callback.interrupted:
        print(f"\nTraining interrupted after {elapsed:.1f}s")
        print_pause_message("scripts/train_combat.py", args.output_dir, model, args.total_timesteps)
        safe_close_vec_envs(train_env, eval_env)
        sys.exit(0)

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

    safe_close_vec_envs(train_env, eval_env)


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
            reward_shaping=False,
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
    parser.add_argument(
        "--load-model", type=str, default=None,
        help="Resume training from a saved MaskablePPO zip (continues timestep count)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume the run in --output-dir from its latest checkpoint",
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=250_000,
        help="Save a resumable checkpoint every N steps (default: 250000)",
    )
    parser.add_argument(
        "--keep-checkpoints", type=int, default=3,
        help="Number of periodic checkpoints to retain (default: 3)",
    )
    parser.add_argument(
        "--policy", type=str, default="attention", choices=("mlp", "attention", "gnn"),
        help="Policy feature extractor: mlp, attention (default), or gnn",
    )
    parser.add_argument(
        "--d-model", type=int, default=128,
        help="Attention model dimension (default: 128)",
    )
    parser.add_argument(
        "--n-heads", type=int, default=4,
        help="Attention heads (default: 4)",
    )
    parser.add_argument(
        "--n-layers", type=int, default=2,
        help="Transformer encoder layers (default: 2)",
    )
    parser.add_argument(
        "--features-dim", type=int, default=256,
        help="Pooled feature dimension fed to pi/vf heads (default: 256)",
    )
    parser.add_argument(
        "--reward-shaping", action="store_true", default=True,
        help="Use non-linear HP and combat micro-reward shaping (default: True)",
    )
    parser.add_argument(
        "--no-reward-shaping", action="store_false", dest="reward_shaping",
        help="Sparse terminal reward only (+1 win / -1 loss)",
    )
    parser.add_argument(
        "--hp-steepness", type=float, default=3.0,
        help="Exponential HP penalty steepness (default: 3.0)",
    )
    parser.add_argument(
        "--vulnerable-scale", type=float, default=0.02,
        help="Micro-reward per Vulnerable stack applied to enemies (default: 0.02)",
    )
    parser.add_argument(
        "--weak-scale", type=float, default=0.02,
        help="Micro-reward per Weak stack applied to enemies (default: 0.02)",
    )
    parser.add_argument(
        "--block-scale", type=float, default=0.001,
        help="Micro-reward per HP blocked from enemy attacks (default: 0.001)",
    )
    args = parser.parse_args()

    from sts2_env.training.checkpointing import resolve_resume_args

    resolve_resume_args(args)

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
            encounter_acts, character_id, character_ids, policy=args.policy,
        )

    train(args)


if __name__ == "__main__":
    main()

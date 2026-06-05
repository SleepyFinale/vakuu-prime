"""Train a MaskablePPO meta-policy on the STS2 hierarchical full-run environment.

Combat is delegated to a pre-trained combat MaskablePPO; card rewards, boss
relics, and rest sites can be auto-resolved by heuristics. This script trains
map navigation, shops, and events.

Usage:
    pip install "sts2-rl-agent[train]"
    python scripts/train_combat.py --acts 0,1,2 --output-dir output/combat_ppo_mixed
    python scripts/train_full_run.py --preset phase1 \\
        --combat-model output/combat_ppo_mixed/best_model/best_model.zip

Requires: stable-baselines3, sb3-contrib, torch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_COMBAT_MODEL = "output/combat_ppo_mixed/best_model/best_model.zip"
DEFAULT_MAX_STEPS = 10_000
DEFAULT_TOTAL_TIMESTEPS = 2_000_000

PRESETS = {
    "phase1": {"act_count": 1, "total_timesteps": 2_000_000},
    "phase2": {"act_count": 3, "total_timesteps": 5_000_000},
    "full": {"act_count": 3, "total_timesteps": 8_000_000},
}


def _resolve_combat_paths(args) -> tuple[
    str | None, dict[int, str] | None, dict[str, str] | None,
]:
    combat_models_by_character = None
    if args.combat_models_by_character:
        from sts2_env.gym_env.hierarchical_run_env import (
            parse_combat_models_by_character,
        )

        parsed = parse_combat_models_by_character(args.combat_models_by_character)
        combat_models_by_character = {
            char_id: str(path) for char_id, path in parsed.items()
        }
        return None, None, combat_models_by_character
    if args.combat_models:
        from sts2_env.gym_env.hierarchical_run_env import parse_combat_models_spec

        parsed = parse_combat_models_spec(args.combat_models)
        return None, {act: str(path) for act, path in parsed.items()}, None
    combat_model = args.combat_model
    if combat_model is None and args.delegate_combat:
        combat_model = DEFAULT_COMBAT_MODEL
    return combat_model, None, None


def _resolve_run_characters(args) -> tuple[str, tuple[str, ...] | None]:
    if args.characters is not None:
        from sts2_env.characters.all import parse_character_ids

        return "Ironclad", parse_character_ids(args.characters)
    return args.character, None


def _validate_combat_paths(
    combat_model: str | None,
    combat_models: dict[int, str] | None,
    combat_models_by_character: dict[str, str] | None,
    delegate_combat: bool,
) -> None:
    if not delegate_combat:
        return
    paths: list[str] = []
    if combat_models_by_character:
        paths.extend(combat_models_by_character.values())
    elif combat_models:
        paths.extend(combat_models.values())
    elif combat_model:
        paths.append(combat_model)
    for path in paths:
        if not Path(path).exists():
            print(f"Combat model not found: {path}")
            print("Train combat first: python scripts/train_combat.py --acts 0,1,2")
            sys.exit(1)


def apply_preset(args) -> None:
    if not args.preset:
        return
    preset = PRESETS.get(args.preset)
    if preset is None:
        print(f"Unknown preset: {args.preset!r} (choices: {', '.join(PRESETS)})")
        sys.exit(1)
    if args.act_count != 1:
        print(f"Preset {args.preset!r} overrides --act-count -> {preset['act_count']}")
    if args.total_timesteps != DEFAULT_TOTAL_TIMESTEPS:
        print(
            f"Preset {args.preset!r} overrides --total-timesteps "
            f"-> {preset['total_timesteps']}"
        )
    args.act_count = preset["act_count"]
    args.total_timesteps = preset["total_timesteps"]
    if args.preset == "phase2" and not args.load_model:
        print(
            "Warning: --preset phase2 is intended for fine-tuning; pass --load-model "
            "(or --resume to continue an existing run in the same --output-dir)"
        )


def make_masked_env(
    seed: int,
    *,
    act_count: int = 1,
    reward_shaping: bool = True,
    combat_model: str | None = None,
    combat_models: dict[int, str] | None = None,
    combat_models_by_character: dict[str, str] | None = None,
    character_id: str = "Ironclad",
    character_ids: tuple[str, ...] | None = None,
    delegate_combat: bool = True,
    use_noncombat_heuristic: bool = True,
    card_value_model: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    act1_biome: str = "random",
    underdocks_unlocked: bool = True,
    underdocks_discovered: bool = True,
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
                combat_models=combat_models,
                combat_models_by_character=combat_models_by_character,
                delegate_combat=True,
                use_noncombat_heuristic=use_noncombat_heuristic,
                card_value_model_path=card_value_model,
                character_id=character_id,
                character_ids=character_ids,
                act_count=act_count,
                reward_shaping=reward_shaping,
                max_steps=max_steps,
                act1_biome=act1_biome,
                underdocks_unlocked=underdocks_unlocked,
                underdocks_discovered=underdocks_discovered,
            )
        else:
            from sts2_env.gym_env.run_env import STS2RunEnv

            env = STS2RunEnv(
                character_id=character_id,
                character_ids=character_ids,
                act_count=act_count,
                reward_shaping=reward_shaping,
                max_steps=max_steps,
                act1_biome=act1_biome,
                underdocks_unlocked=underdocks_unlocked,
                underdocks_discovered=underdocks_discovered,
            )
        env = ActionMasker(env, mask_fn)
        return env

    return _init


def train(args):
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        from sts2_env.training.callbacks import RunWinRateCallback
    except ImportError:
        print("Training requires sb3-contrib and stable-baselines3.")
        print("Install with: pip install 'sts2-rl-agent[train]'")
        sys.exit(1)

    from sts2_env.training.checkpointing import (
        build_ppo_callbacks,
        print_pause_message,
        print_resume_progress,
        save_run_config,
    )

    combat_model, combat_models, combat_models_by_character = _resolve_combat_paths(args)
    character_id, character_ids = _resolve_run_characters(args)
    _validate_combat_paths(
        combat_model,
        combat_models,
        combat_models_by_character,
        args.delegate_combat,
    )

    print("Training MaskablePPO on STS2 Hierarchical Full Run")
    if args.preset:
        print(f"  preset:           {args.preset}")
    print(f"  act_count:        {args.act_count}")
    if character_ids is not None:
        print(f"  characters:       {character_ids}")
    else:
        print(f"  character:        {character_id}")
    print(f"  n_envs:           {args.n_envs}")
    print(f"  total_timesteps:  {args.total_timesteps}")
    print(f"  learning_rate:    {args.lr}")
    print(f"  batch_size:       {args.batch_size}")
    print(f"  reward_shaping:   {args.reward_shaping}")
    print(f"  delegate_combat:  {args.delegate_combat}")
    print(f"  noncombat_heur:   {args.use_noncombat_heuristic}")
    if combat_models_by_character:
        print(f"  combat_by_char:   {combat_models_by_character}")
    elif combat_models:
        print(f"  combat_models:    {combat_models}")
    else:
        print(f"  combat_model:     {combat_model}")
    if args.card_value_model:
        print(f"  card_value_model: {args.card_value_model}")
    print(f"  max_steps:        {args.max_steps}")
    print(f"  output_dir:       {args.output_dir}")
    if args.load_model:
        print(f"  load_model:       {args.load_model}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not getattr(args, "resume", False):
        save_run_config(output_dir, args)

    env_kwargs = dict(
        act_count=args.act_count,
        reward_shaping=args.reward_shaping,
        combat_model=combat_model,
        combat_models=combat_models,
        combat_models_by_character=combat_models_by_character,
        character_id=character_id,
        character_ids=character_ids,
        delegate_combat=args.delegate_combat,
        use_noncombat_heuristic=args.use_noncombat_heuristic,
        card_value_model=args.card_value_model,
        max_steps=args.max_steps,
        act1_biome=args.act1_biome,
        underdocks_unlocked=args.underdocks_unlocked,
        underdocks_discovered=args.underdocks_discovered,
    )

    if args.n_envs > 1:
        train_env = SubprocVecEnv([
            make_masked_env(i, **env_kwargs)
            for i in range(args.n_envs)
        ])
    else:
        train_env = DummyVecEnv([make_masked_env(0, **env_kwargs)])

    eval_env = DummyVecEnv([make_masked_env(9999, **env_kwargs)])

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
            verbose=1,
            tensorboard_log=str(output_dir / "tb_logs"),
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
        )
        reset_timesteps = True

    # RunWinRateCallback expects a Gymnasium-style env (reset(seed=...) -> (obs, info),
    # step(...) -> obs, reward, terminated, truncated, info). EvalCallback operates
    # on the vectorized env, but the win-rate callback should use the single wrapped
    # environment instance directly to avoid VecEnv API mismatches.
    win_rate_callback = RunWinRateCallback(
        eval_env.envs[0],
        eval_freq=max((args.eval_freq * 5) // args.n_envs, 1),
        n_eval_episodes=max(args.eval_episodes // 2, 1),
    )
    callbacks, interrupt_callback = build_ppo_callbacks(
        output_dir=output_dir,
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        n_envs=args.n_envs,
        checkpoint_freq=args.checkpoint_freq,
        keep_checkpoints=args.keep_checkpoints,
        extra_callbacks=(win_rate_callback,),
    )

    start = time.perf_counter()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=reset_timesteps,
    )
    elapsed = time.perf_counter() - start

    if interrupt_callback.interrupted:
        print(f"\nTraining interrupted after {elapsed:.1f}s")
        print_pause_message("scripts/train_full_run.py", args.output_dir, model, args.total_timesteps)
        train_env.close()
        eval_env.close()
        return True

    final_path = str(output_dir / "final_model")
    model.save(final_path)
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {output_dir / 'best_model'}")

    print("\n--- Final Evaluation (sparse rewards, no heuristics) ---")
    evaluate(
        model,
        act_count=args.act_count,
        combat_model=combat_model,
        combat_models=combat_models,
        combat_models_by_character=combat_models_by_character,
        character_id=character_id,
        character_ids=character_ids,
        delegate_combat=args.delegate_combat,
        card_value_model=args.card_value_model,
        use_noncombat_heuristic=False,
        n_episodes=100,
    )

    train_env.close()
    eval_env.close()
    return False


def _make_eval_env(
    act_count: int,
    combat_model: str | None,
    combat_models: dict[int, str] | None,
    combat_models_by_character: dict[str, str] | None,
    character_id: str,
    character_ids: tuple[str, ...] | None,
    delegate_combat: bool,
    card_value_model: str | None = None,
    use_noncombat_heuristic: bool = False,
    act1_biome: str = "random",
    underdocks_unlocked: bool = True,
    underdocks_discovered: bool = True,
):
    from sb3_contrib.common.wrappers import ActionMasker

    if delegate_combat:
        from sts2_env.gym_env.hierarchical_run_env import STS2HierarchicalRunEnv

        base = STS2HierarchicalRunEnv(
            combat_model_path=combat_model or DEFAULT_COMBAT_MODEL,
            combat_models=combat_models,
            combat_models_by_character=combat_models_by_character,
            delegate_combat=True,
            use_noncombat_heuristic=use_noncombat_heuristic,
            card_value_model_path=card_value_model,
            character_id=character_id,
            character_ids=character_ids,
            act_count=act_count,
            reward_shaping=False,
            max_steps=DEFAULT_MAX_STEPS,
            act1_biome=act1_biome,
            underdocks_unlocked=underdocks_unlocked,
            underdocks_discovered=underdocks_discovered,
        )
    else:
        from sts2_env.gym_env.run_env import STS2RunEnv

        base = STS2RunEnv(
            character_id=character_id,
            character_ids=character_ids,
            act_count=act_count,
            reward_shaping=False,
            max_steps=DEFAULT_MAX_STEPS,
            act1_biome=act1_biome,
            underdocks_unlocked=underdocks_unlocked,
            underdocks_discovered=underdocks_discovered,
        )

    def mask_fn(env):
        return env.action_masks()

    return ActionMasker(base, mask_fn)


def evaluate(
    model,
    act_count: int = 1,
    combat_model: str | None = None,
    combat_models: dict[int, str] | None = None,
    combat_models_by_character: dict[str, str] | None = None,
    character_id: str = "Ironclad",
    character_ids: tuple[str, ...] | None = None,
    delegate_combat: bool = True,
    card_value_model: str | None = None,
    use_noncombat_heuristic: bool = False,
    n_episodes: int = 100,
    act1_biome: str = "random",
    underdocks_unlocked: bool = True,
    underdocks_discovered: bool = True,
):
    """Evaluate with sparse terminal rewards."""
    env = _make_eval_env(
        act_count,
        combat_model,
        combat_models,
        combat_models_by_character,
        character_id,
        character_ids,
        delegate_combat,
        card_value_model=card_value_model,
        use_noncombat_heuristic=use_noncombat_heuristic,
        act1_biome=act1_biome,
        underdocks_unlocked=underdocks_unlocked,
        underdocks_discovered=underdocks_discovered,
    )

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
    combat_models: dict[int, str] | None = None,
    combat_models_by_character: dict[str, str] | None = None,
    character_id: str = "Ironclad",
    character_ids: tuple[str, ...] | None = None,
    delegate_combat: bool = True,
    use_noncombat_heuristic: bool = False,
    n_episodes: int = 100,
    act1_biome: str = "random",
    underdocks_unlocked: bool = True,
    underdocks_discovered: bool = True,
):
    """Random-action baseline with sparse rewards."""
    if delegate_combat:
        from sts2_env.gym_env.hierarchical_run_env import STS2HierarchicalRunEnv

        env = STS2HierarchicalRunEnv(
            combat_model_path=combat_model or DEFAULT_COMBAT_MODEL,
            combat_models=combat_models,
            combat_models_by_character=combat_models_by_character,
            delegate_combat=True,
            use_noncombat_heuristic=use_noncombat_heuristic,
            character_id=character_id,
            character_ids=character_ids,
            act_count=act_count,
            reward_shaping=False,
            max_steps=DEFAULT_MAX_STEPS,
            act1_biome=act1_biome,
            underdocks_unlocked=underdocks_unlocked,
            underdocks_discovered=underdocks_discovered,
        )
    else:
        from sts2_env.gym_env.run_env import STS2RunEnv

        env = STS2RunEnv(
            character_id=character_id,
            character_ids=character_ids,
            act_count=act_count,
            reward_shaping=False,
            max_steps=DEFAULT_MAX_STEPS,
            act1_biome=act1_biome,
            underdocks_unlocked=underdocks_unlocked,
            underdocks_discovered=underdocks_discovered,
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

    print("=== Random Baseline (sparse eval) ===")
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
        "--preset", type=str, default=None, choices=sorted(PRESETS),
        help="Training preset: phase1 (2M, act1), phase2 (5M, full), full (8M)",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {DEFAULT_TOTAL_TIMESTEPS})",
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
        "--act1-biome", type=str, default="random",
        choices=("random", "overgrowth", "underdocks"),
        help="Act 1 biome: random (game rules), overgrowth, or underdocks",
    )
    parser.add_argument(
        "--underdocks-unlocked", action=argparse.BooleanOptionalAction, default=True,
        help="Whether Underdocks epoch is unlocked (default: True)",
    )
    parser.add_argument(
        "--underdocks-discovered", action=argparse.BooleanOptionalAction, default=True,
        help="Whether Underdocks was discovered on save (False forces first-run Underdocks)",
    )
    parser.add_argument(
        "--combat-model", type=str, default=None,
        help=f"Single combat model zip (default: {DEFAULT_COMBAT_MODEL})",
    )
    parser.add_argument(
        "--combat-models", type=str, default=None,
        help="Per-act combat models: '0:path0,1:path1,2:path2'",
    )
    parser.add_argument(
        "--combat-models-by-character", type=str, default=None,
        help="Per-character combat models: 'Ironclad:path0,Silent:path1'",
    )
    from sts2_env.characters.all import SUPPORTED_TRAINING_CHARACTERS

    character_group = parser.add_mutually_exclusive_group()
    character_group.add_argument(
        "--character", type=str, default="Ironclad",
        choices=SUPPORTED_TRAINING_CHARACTERS,
        help="Character for full-run training (default: Ironclad)",
    )
    character_group.add_argument(
        "--characters", type=str, default=None,
        help="Mixed-character full-run pool: 'Ironclad,Silent' or 'all'",
    )
    parser.add_argument(
        "--no-combat-delegate", action="store_true",
        help="Train flat policy without combat delegation (ablation)",
    )
    parser.add_argument(
        "--no-noncombat-heuristic", action="store_true",
        help="Disable auto card-reward / boss-relic / rest heuristics",
    )
    parser.add_argument(
        "--card-value-model", type=str, default=None,
        help="Learned card-value model (.pt or output dir) for card rewards",
    )
    parser.add_argument(
        "--eval-with-heuristics", action="store_true",
        help="Also evaluate with rule heuristics for card/boss/rest",
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
    args = parser.parse_args()
    args.delegate_combat = not args.no_combat_delegate
    args.use_noncombat_heuristic = not args.no_noncombat_heuristic

    from sts2_env.training.checkpointing import resolve_resume_args

    resuming = resolve_resume_args(args)
    if not resuming:
        apply_preset(args)

    combat_model, combat_models, combat_models_by_character = _resolve_combat_paths(args)
    character_id, character_ids = _resolve_run_characters(args)

    if args.baseline_only:
        random_baseline(
            act_count=args.act_count,
            combat_model=combat_model,
            combat_models=combat_models,
            combat_models_by_character=combat_models_by_character,
            character_id=character_id,
            character_ids=character_ids,
            delegate_combat=args.delegate_combat,
            use_noncombat_heuristic=args.use_noncombat_heuristic,
        )
    else:
        if args.delegate_combat:
            _validate_combat_paths(
                combat_model,
                combat_models,
                combat_models_by_character,
                True,
            )
        print("Running random baseline for reference...")
        random_baseline(
            act_count=args.act_count,
            combat_model=combat_model,
            combat_models=combat_models,
            combat_models_by_character=combat_models_by_character,
            character_id=character_id,
            character_ids=character_ids,
            delegate_combat=args.delegate_combat,
            use_noncombat_heuristic=False,
            n_episodes=50,
        )
        print()
        interrupted = train(args)
        if not interrupted and args.eval_with_heuristics:
            print("\n--- Heuristic-Assisted Evaluation ---")
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(str(Path(args.output_dir) / "final_model"))
            evaluate(
                model,
                act_count=args.act_count,
                combat_model=combat_model,
                combat_models=combat_models,
                combat_models_by_character=combat_models_by_character,
                character_id=character_id,
                character_ids=character_ids,
                delegate_combat=args.delegate_combat,
                use_noncombat_heuristic=True,
                n_episodes=50,
            )


if __name__ == "__main__":
    main()

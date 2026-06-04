"""Evaluate a trained full-run meta-policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_COMBAT_MODEL = "output/combat_ppo_mixed/best_model/best_model.zip"
DEFAULT_MAX_STEPS = 10_000


def main():
    parser = argparse.ArgumentParser(description="Evaluate STS2 full-run meta-policy")
    parser.add_argument("--load-model", type=str, required=True, help="Meta-policy zip")
    parser.add_argument("--combat-model", type=str, default=None)
    parser.add_argument("--combat-models", type=str, default=None)
    parser.add_argument("--card-value-model", type=str, default=None)
    parser.add_argument("--act-count", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--with-heuristics", action="store_true",
        help="Enable rule heuristics for card/boss/rest",
    )
    args = parser.parse_args()

    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from sts2_env.gym_env.hierarchical_run_env import (
            STS2HierarchicalRunEnv,
            parse_combat_models_spec,
        )
    except ImportError:
        print("Requires sb3-contrib. pip install 'sts2-rl-agent[train]'")
        sys.exit(1)

    combat_model = args.combat_model or DEFAULT_COMBAT_MODEL
    combat_models = None
    if args.combat_models:
        combat_models = {
            act: str(path)
            for act, path in parse_combat_models_spec(args.combat_models).items()
        }

    def mask_fn(env):
        return env.action_masks()

    env = ActionMasker(
        STS2HierarchicalRunEnv(
            combat_model_path=combat_model,
            combat_models=combat_models,
            delegate_combat=True,
            use_noncombat_heuristic=args.with_heuristics,
            card_value_model_path=args.card_value_model,
            act_count=args.act_count,
            reward_shaping=False,
            max_steps=args.max_steps,
        ),
        mask_fn,
    )

    model = MaskablePPO.load(args.load_model, env=env)

    wins = 0
    floors = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep + 20_000)
        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            if terminated and reward > 0:
                wins += 1
        unwrapped = env.unwrapped
        floors.append(
            unwrapped.run_state.total_floor
            if unwrapped.run_state is not None
            else 0
        )

    print(f"Model:           {args.load_model}")
    print(f"Episodes:        {args.episodes}")
    print(f"Act count:       {args.act_count}")
    print(f"Heuristics:      {args.with_heuristics}")
    print(f"Card-value:      {args.card_value_model}")
    print(f"Win rate:        {wins / args.episodes:.1%}")
    print(f"Avg floors:      {np.mean(floors):.1f}")
    print(f"Max floors:      {max(floors)}")


if __name__ == "__main__":
    main()

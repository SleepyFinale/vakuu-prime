"""Stable-Baselines3 callbacks for full-run training."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class RunWinRateCallback(BaseCallback):
    """Log full-run win rate, floors, and meta-step length during training."""

    def __init__(
        self,
        eval_env,
        eval_freq: int = 20_000,
        n_eval_episodes: int = 10,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = max(eval_freq, 1)
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        wins = 0
        floors: list[float] = []
        meta_steps: list[float] = []

        for ep in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset(seed=10_000 + ep)
            done = False
            steps = 0
            while not done:
                import numpy as np

                mask = self.eval_env.action_masks()
                action, _ = self.model.predict(
                    obs,
                    action_masks=mask,
                    deterministic=True,
                )
                obs, reward, terminated, truncated, info = self.eval_env.step(int(action))
                steps += 1
                done = terminated or truncated
                if terminated and reward > 0:
                    wins += 1
            floors.append(float(info.get("floor", 0)))
            meta_steps.append(float(info.get("meta_step", steps)))
            unwrapped = self.eval_env.unwrapped
            if hasattr(unwrapped, "run_state") and unwrapped.run_state is not None:
                floors[-1] = float(unwrapped.run_state.total_floor)

        win_rate = wins / max(self.n_eval_episodes, 1)
        mean_floors = sum(floors) / max(len(floors), 1)
        mean_meta = sum(meta_steps) / max(len(meta_steps), 1)

        self.logger.record("eval/win_rate", win_rate)
        self.logger.record("eval/mean_floors", mean_floors)
        self.logger.record("eval/mean_meta_steps", mean_meta)

        if self.verbose:
            print(
                f"[RunWinRate] win_rate={win_rate:.1%}  "
                f"floors={mean_floors:.1f}  meta_steps={mean_meta:.1f}"
            )
            if win_rate == 0.0 and self.num_timesteps >= 500_000:
                print(
                    "Hint: consider --card-value-model, --preset full (8M steps), "
                    "or mixed-act combat training."
                )
        return True

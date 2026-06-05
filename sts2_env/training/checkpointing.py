"""Pause/resume helpers for the PPO training scripts.

Stable-Baselines3 already stores policy weights, optimizer state, and the
timestep counter inside each ``.zip``. Resuming therefore only requires loading
the latest zip and calling ``learn(..., reset_num_timesteps=False)`` until the
original ``total_timesteps`` target is reached. This module adds the missing
pieces: periodic checkpoints, graceful Ctrl+C handling, a persisted run config,
and a ``--resume`` flag that locates the most recent checkpoint automatically.
"""

from __future__ import annotations

import json
import re
import signal
import sys
from datetime import datetime
from pathlib import Path

_STEP_RE = re.compile(r"_(\d+)_steps\.zip$")
_JSON_SCALARS = (str, int, float, bool, type(None))


def _checkpoint_steps(path: Path) -> int:
    match = _STEP_RE.search(path.name)
    return int(match.group(1)) if match else -1


def save_run_config(output_dir: str | Path, args) -> None:
    """Persist the resolved CLI args so a later ``--resume`` can replay them."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_args = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, _JSON_SCALARS)
    }
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "args": saved_args,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def load_run_config(output_dir: str | Path) -> dict | None:
    """Read ``run_config.json`` from a previous run, or ``None`` if absent."""
    path = Path(output_dir) / "run_config.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def find_latest_ppo_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the best checkpoint to resume from, by priority.

    Order: an ``interrupted_checkpoint.zip`` (most recent progress) > the newest
    numbered periodic checkpoint > ``final_model.zip`` > the eval ``best_model``.
    """
    output_dir = Path(output_dir)
    interrupted = output_dir / "interrupted_checkpoint.zip"
    if interrupted.exists():
        return interrupted
    ckpt_dir = output_dir / "checkpoints"
    if ckpt_dir.is_dir():
        numbered = sorted(
            ckpt_dir.glob("checkpoint_*_steps.zip"), key=_checkpoint_steps
        )
        if numbered:
            return numbered[-1]
    final = output_dir / "final_model.zip"
    if final.exists():
        return final
    best = output_dir / "best_model" / "best_model.zip"
    if best.exists():
        return best
    return None


def prune_old_checkpoints(
    directory: str | Path, name_prefix: str = "checkpoint", keep: int = 3
) -> None:
    """Delete all but the ``keep`` newest numbered checkpoints in ``directory``."""
    if keep <= 0:
        return
    directory = Path(directory)
    if not directory.is_dir():
        return
    numbered = sorted(
        directory.glob(f"{name_prefix}_*_steps.zip"), key=_checkpoint_steps
    )
    for old in numbered[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


def _flag_in_argv(flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


def _restore_unset_args(args, config: dict) -> None:
    """Overwrite args the user did not pass on the resume command line."""
    saved_args = config.get("args", {})
    for key, value in saved_args.items():
        if key in ("resume", "load_model"):
            continue
        dashed = key.replace("_", "-")
        if _flag_in_argv(f"--{dashed}") or _flag_in_argv(f"--no-{dashed}"):
            continue
        if hasattr(args, key):
            setattr(args, key, value)


def resolve_resume_args(args) -> bool:
    """Wire up ``--resume``: restore saved args and locate the latest checkpoint.

    Returns ``True`` when resuming so the caller can skip re-saving the run
    config and skip preset re-application. Exits with a clear message when the
    request cannot be satisfied.
    """
    if not getattr(args, "resume", False):
        return False
    if getattr(args, "load_model", None):
        print("Error: --resume and --load-model are mutually exclusive.")
        sys.exit(1)
    if not args.output_dir:
        print("Error: --resume requires --output-dir to locate the checkpoint.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    config = load_run_config(output_dir)
    if config is not None:
        _restore_unset_args(args, config)

    checkpoint = find_latest_ppo_checkpoint(output_dir)
    if checkpoint is None:
        print(f"No checkpoint found in {output_dir} to resume from.")
        print("Start a fresh run without --resume.")
        sys.exit(1)

    args.load_model = str(checkpoint)
    print(f"Resuming from checkpoint: {checkpoint}")
    return True


def print_resume_progress(model, total_timesteps: int) -> None:
    print(
        f"  Loaded at {model.num_timesteps:,} timesteps "
        f"(target {total_timesteps:,})"
    )


def print_pause_message(script: str, output_dir: str | Path, model, total_timesteps: int) -> None:
    print(
        f"\nTraining paused at {model.num_timesteps:,} / "
        f"{total_timesteps:,} timesteps."
    )
    print("Resume with:")
    print(f"  python {script} --resume --output-dir {output_dir}")


def build_ppo_callbacks(
    *,
    output_dir: str | Path,
    eval_env,
    eval_freq: int,
    eval_episodes: int,
    n_envs: int,
    checkpoint_freq: int,
    keep_checkpoints: int,
    extra_callbacks=(),
):
    """Build the callback list shared by the PPO trainers.

    Returns ``(callback_list, interrupt_callback)``; the interrupt callback
    exposes ``.interrupted`` so the caller can detect a paused run after
    ``learn()`` returns. SB3 is imported lazily here so the pure-Python helpers
    above stay importable without the optional ``[train]`` dependencies.
    """
    from stable_baselines3.common.callbacks import CallbackList, EvalCallback

    output_dir = Path(output_dir)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=eval_episodes,
        deterministic=False,
    )
    checkpoint_callback = _build_pruning_checkpoint_callback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=str(output_dir / "checkpoints"),
        keep=keep_checkpoints,
    )
    interrupt_callback = _build_interrupt_callback(
        save_path=str(output_dir / "interrupted_checkpoint"),
    )
    callbacks = [eval_callback, checkpoint_callback, *extra_callbacks, interrupt_callback]
    return CallbackList(callbacks), interrupt_callback


def _build_pruning_checkpoint_callback(*, save_freq: int, save_path: str, keep: int):
    from stable_baselines3.common.callbacks import CheckpointCallback

    class _PruningCheckpointCallback(CheckpointCallback):
        """CheckpointCallback that keeps only the newest ``keep`` checkpoints."""

        def _on_step(self) -> bool:
            result = super()._on_step()
            if self.n_calls % self.save_freq == 0:
                prune_old_checkpoints(Path(self.save_path), self.name_prefix, keep)
            return result

    return _PruningCheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        name_prefix="checkpoint",
        verbose=1,
    )


def _build_interrupt_callback(*, save_path: str):
    from stable_baselines3.common.callbacks import BaseCallback

    class StopTrainingOnInterrupt(BaseCallback):
        """Save a checkpoint and stop ``learn()`` cleanly on SIGINT/SIGTERM."""

        def __init__(self, path: str | Path, verbose: int = 1):
            super().__init__(verbose)
            self.save_path = Path(path)
            self.interrupted = False
            self._original_handlers: dict = {}

        def _handle_signal(self, signum, frame) -> None:
            self.interrupted = True
            self._restore_handlers()
            if self.verbose:
                print(
                    "\nInterrupt received; saving checkpoint and stopping "
                    "after the current step (press again to force-quit)..."
                )

        def _install_handlers(self) -> None:
            for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
                if sig is None:
                    continue
                try:
                    self._original_handlers[sig] = signal.signal(sig, self._handle_signal)
                except (ValueError, OSError, RuntimeError):
                    pass

        def _restore_handlers(self) -> None:
            for sig, original in self._original_handlers.items():
                try:
                    signal.signal(sig, original)
                except (ValueError, OSError, RuntimeError):
                    pass
            self._original_handlers.clear()

        def _on_training_start(self) -> None:
            self.interrupted = False
            self._install_handlers()

        def _on_step(self) -> bool:
            if self.interrupted:
                self.model.save(str(self.save_path))
                if self.verbose:
                    print(f"Saved interrupted checkpoint to {self.save_path}.zip")
                return False
            return True

        def _on_training_end(self) -> None:
            self._restore_handlers()

    return StopTrainingOnInterrupt(path=save_path)

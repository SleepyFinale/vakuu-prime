"""Tests for training pause/resume helpers (no full PPO training required)."""

import sys
from types import SimpleNamespace

import pytest

from sts2_env.training import checkpointing as ckpt


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_find_latest_prefers_interrupted(tmp_path):
    _touch(tmp_path / "final_model.zip")
    _touch(tmp_path / "best_model" / "best_model.zip")
    _touch(tmp_path / "checkpoints" / "checkpoint_1000_steps.zip")
    _touch(tmp_path / "interrupted_checkpoint.zip")
    assert (
        ckpt.find_latest_ppo_checkpoint(tmp_path)
        == tmp_path / "interrupted_checkpoint.zip"
    )


def test_find_latest_returns_newest_numbered_checkpoint(tmp_path):
    _touch(tmp_path / "final_model.zip")
    _touch(tmp_path / "checkpoints" / "checkpoint_1000_steps.zip")
    _touch(tmp_path / "checkpoints" / "checkpoint_250000_steps.zip")
    _touch(tmp_path / "checkpoints" / "checkpoint_50000_steps.zip")
    assert (
        ckpt.find_latest_ppo_checkpoint(tmp_path)
        == tmp_path / "checkpoints" / "checkpoint_250000_steps.zip"
    )


def test_find_latest_fallback_order(tmp_path):
    assert ckpt.find_latest_ppo_checkpoint(tmp_path) is None
    _touch(tmp_path / "best_model" / "best_model.zip")
    assert (
        ckpt.find_latest_ppo_checkpoint(tmp_path)
        == tmp_path / "best_model" / "best_model.zip"
    )
    _touch(tmp_path / "final_model.zip")
    assert ckpt.find_latest_ppo_checkpoint(tmp_path) == tmp_path / "final_model.zip"


def test_prune_keeps_only_newest(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    for steps in (1000, 2000, 3000, 4000, 5000):
        _touch(ckpt_dir / f"checkpoint_{steps}_steps.zip")
    ckpt.prune_old_checkpoints(ckpt_dir, keep=3)
    remaining = sorted(p.name for p in ckpt_dir.glob("checkpoint_*_steps.zip"))
    assert remaining == [
        "checkpoint_3000_steps.zip",
        "checkpoint_4000_steps.zip",
        "checkpoint_5000_steps.zip",
    ]


def test_resolve_resume_restores_saved_args(tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    ckpt.save_run_config(
        output_dir,
        SimpleNamespace(
            output_dir=str(output_dir),
            total_timesteps=4_000_000,
            acts="0,1,2",
            resume=False,
            load_model=None,
        ),
    )
    _touch(output_dir / "final_model.zip")

    monkeypatch.setattr(
        sys, "argv", ["train_combat.py", "--resume", "--output-dir", str(output_dir)]
    )
    args = SimpleNamespace(
        output_dir=str(output_dir),
        total_timesteps=None,
        acts="0",
        resume=True,
        load_model=None,
    )
    resuming = ckpt.resolve_resume_args(args)

    assert resuming is True
    assert args.load_model == str(output_dir / "final_model.zip")
    assert args.total_timesteps == 4_000_000
    assert args.acts == "0,1,2"


def test_resolve_resume_keeps_user_override(tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    ckpt.save_run_config(
        output_dir,
        SimpleNamespace(
            output_dir=str(output_dir),
            total_timesteps=4_000_000,
            resume=False,
            load_model=None,
        ),
    )
    _touch(output_dir / "final_model.zip")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_combat.py",
            "--resume",
            "--output-dir",
            str(output_dir),
            "--total-timesteps",
            "6000000",
        ],
    )
    args = SimpleNamespace(
        output_dir=str(output_dir),
        total_timesteps=6_000_000,
        resume=True,
        load_model=None,
    )
    ckpt.resolve_resume_args(args)

    assert args.total_timesteps == 6_000_000


def test_resolve_resume_rejects_load_model_combo(monkeypatch, tmp_path):
    args = SimpleNamespace(
        output_dir=str(tmp_path), resume=True, load_model="some/model.zip"
    )
    with pytest.raises(SystemExit):
        ckpt.resolve_resume_args(args)


def test_resolve_resume_exits_without_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys, "argv", ["train_combat.py", "--resume", "--output-dir", str(tmp_path)]
    )
    args = SimpleNamespace(output_dir=str(tmp_path), resume=True, load_model=None)
    with pytest.raises(SystemExit):
        ckpt.resolve_resume_args(args)


def test_safe_close_vec_envs_swallows_broken_pipe():
    class BrokenEnv:
        def close(self):
            raise BrokenPipeError("pipe closed")

    class GoodEnv:
        closed = False

        def close(self):
            GoodEnv.closed = True

    ckpt.safe_close_vec_envs(BrokenEnv(), GoodEnv())
    assert GoodEnv.closed is True


def test_save_interrupted_checkpoint_is_idempotent(tmp_path):
    state = ckpt.InterruptState()
    saved = []

    model = SimpleNamespace(save=lambda path: saved.append(path))

    path = tmp_path / "interrupted_checkpoint"
    assert ckpt.save_interrupted_checkpoint(model, path, state, verbose=0) is True
    assert state.saved is True
    assert saved == [str(path)]
    assert ckpt.save_interrupted_checkpoint(model, path, state, verbose=0) is False
    assert len(saved) == 1


def test_handle_training_keyboard_interrupt_saves_once(tmp_path):
    state = ckpt.InterruptState()
    saved = []

    model = SimpleNamespace(save=lambda path: saved.append(path))
    interrupt_callback = SimpleNamespace(
        interrupt_state=state,
        save_path=tmp_path / "interrupted_checkpoint",
        verbose=0,
    )

    ckpt.handle_training_keyboard_interrupt(model, interrupt_callback)
    assert state.requested is True
    assert state.saved is True
    assert len(saved) == 1

    ckpt.handle_training_keyboard_interrupt(model, interrupt_callback)
    assert len(saved) == 1


def test_interrupt_signal_handler_sets_requested():
    pytest.importorskip("stable_baselines3")
    state = ckpt.InterruptState()
    interrupt_callback = ckpt._build_interrupt_callback(
        save_path="ignored",
        interrupt_state=state,
    )
    interrupt_callback.verbose = 0
    interrupt_callback._handle_signal(None, None)
    assert state.requested is True
    assert interrupt_callback.interrupted is True


def test_interrupt_second_signal_force_quits():
    pytest.importorskip("stable_baselines3")
    state = ckpt.InterruptState(requested=True)
    interrupt_callback = ckpt._build_interrupt_callback(
        save_path="ignored",
        interrupt_state=state,
    )
    interrupt_callback.verbose = 0
    with pytest.raises(SystemExit) as exc:
        interrupt_callback._handle_signal(None, None)
    assert exc.value.code == 130


def test_interrupt_callback_saves_on_step(tmp_path):
    pytest.importorskip("stable_baselines3")
    state = ckpt.InterruptState(requested=True)
    saved = []

    interrupt_callback = ckpt._build_interrupt_callback(
        save_path=tmp_path / "interrupted_checkpoint",
        interrupt_state=state,
    )
    interrupt_callback.model = SimpleNamespace(
        save=lambda path: saved.append(path)
    )
    interrupt_callback.verbose = 0

    assert interrupt_callback._on_step() is False
    assert saved == [str(tmp_path / "interrupted_checkpoint")]
    assert state.saved is True


def _minimal_eval_vec_env():
    """Tiny VecEnv so EvalCallback can be constructed in unit tests."""
    pytest.importorskip("stable_baselines3")
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    class _TrivialEnv(gym.Env):
        observation_space = gym.spaces.Box(0.0, 1.0, (1,), dtype=float)
        action_space = gym.spaces.Discrete(1)

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            obs = self.observation_space.sample()
            return obs, 0.0, True, False, {}

    return DummyVecEnv([_TrivialEnv])


def test_interruptible_eval_skips_when_already_requested():
    state = ckpt.InterruptState(requested=True)
    eval_callback = ckpt._build_interruptible_eval_callback(
        eval_env=_minimal_eval_vec_env(),
        interrupt_state=state,
        eval_freq=1,
        n_eval_episodes=1,
        log_path=None,
        best_model_save_path=None,
    )
    eval_callback.eval_freq = 1
    eval_callback.n_calls = 1

    assert eval_callback._on_step() is False


def test_interruptible_eval_aborts_mid_eval(monkeypatch):
    from stable_baselines3.common import callbacks as sb3_callbacks

    state = ckpt.InterruptState()

    def fake_evaluate_policy(*args, **kwargs):
        callback = kwargs.get("callback")
        assert callback is not None
        callback({"done": False, "info": {}}, {})
        state.requested = True
        callback({"done": False, "info": {}}, {})
        return ([1.0], [10])

    monkeypatch.setattr(sb3_callbacks, "evaluate_policy", fake_evaluate_policy)

    eval_callback = ckpt._build_interruptible_eval_callback(
        eval_env=_minimal_eval_vec_env(),
        interrupt_state=state,
        eval_freq=1,
        n_eval_episodes=2,
        log_path=None,
        best_model_save_path=None,
    )
    eval_callback.eval_freq = 1
    eval_callback.n_calls = 1
    eval_callback.model = SimpleNamespace(get_vec_normalize_env=lambda: None)

    assert eval_callback._on_step() is False


def test_card_value_training_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from sts2_env.gym_env.card_value import (
        CardValueConfig,
        build_card_value_net,
        load_training_checkpoint,
        save_training_checkpoint,
    )

    config = CardValueConfig()
    model = build_card_value_net(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    path = tmp_path / "training_checkpoint.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=4,
        best_val_acc=0.75,
        config=config,
    )

    data = load_training_checkpoint(path)
    assert data["epoch"] == 4
    assert data["best_val_acc"] == 0.75
    assert data["config"]["hidden_size"] == config.hidden_size

    fresh = build_card_value_net(config)
    fresh.load_state_dict(data["model_state"])

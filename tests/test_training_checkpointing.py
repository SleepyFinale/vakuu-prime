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

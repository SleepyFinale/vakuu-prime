"""Collect card-reward data and train an outcome-weighted card-value network.

Usage:
    python scripts/train_card_value.py --collect-episodes 2000
    python scripts/train_card_value.py --data output/card_value_data/episodes.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_COMBAT_MODEL = "output/combat_ppo_mixed/best_model/best_model.zip"
DEFAULT_DATA_PATH = "output/card_value_data/episodes.npz"
DEFAULT_OUTPUT_DIR = "output/card_value"


def _partial_path(data_path: Path) -> Path:
    return data_path.with_name(data_path.stem + ".partial.npz")


def collect_episodes(
    n_episodes: int,
    data_path: Path,
    combat_model: str,
    character_id: str = "Ironclad",
    seed: int = 0,
    loss_weight: float = 0.3,
    resume: bool = False,
) -> None:
    from sts2_env.gym_env.card_value import (
        encode_card_reward_sample,
        label_from_rules,
    )
    from sts2_env.gym_env.hierarchical_run_env import STS2HierarchicalRunEnv
    from sts2_env.run.run_manager import RunManager

    if not Path(combat_model).exists():
        print(f"Combat model not found: {combat_model}")
        sys.exit(1)

    contexts: list[np.ndarray] = []
    card_features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    pending_indices: list[int] = []

    partial_path = _partial_path(data_path)
    start_ep = 0
    if resume and partial_path.exists():
        partial = np.load(partial_path)
        contexts = list(partial["contexts"])
        card_features = list(partial["card_features"])
        masks = list(partial["masks"])
        labels = list(partial["labels"])
        weights = list(partial["weights"])
        start_ep = int(partial["episodes_done"])
        print(
            f"Resuming collection from episode {start_ep}/{n_episodes} "
            f"({len(labels)} samples already collected)"
        )
        if start_ep >= n_episodes:
            print("Partial dataset already has the requested episode count.")

    def observe_card_reward(mgr: RunManager) -> None:
        if not mgr._offered_cards:
            return
        context, cards, card_mask, _ = encode_card_reward_sample(mgr)
        label = label_from_rules(mgr)
        contexts.append(context)
        card_features.append(cards)
        masks.append(card_mask)
        labels.append(label)
        weights.append(1.0)
        pending_indices.append(len(labels) - 1)

    env = STS2HierarchicalRunEnv(
        combat_model_path=combat_model,
        delegate_combat=True,
        use_noncombat_heuristic=True,
        character_id=character_id,
        act_count=3,
        reward_shaping=False,
        card_reward_observer=observe_card_reward,
    )

    def save_partial(episodes_done: int) -> None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            partial_path,
            contexts=np.stack(contexts) if contexts else np.empty((0,)),
            card_features=np.stack(card_features) if card_features else np.empty((0,)),
            masks=np.stack(masks) if masks else np.empty((0,)),
            labels=np.array(labels, dtype=np.int64),
            weights=np.array(weights, dtype=np.float32),
            episodes_done=np.int64(episodes_done),
        )

    for ep in range(start_ep, n_episodes):
        rng = np.random.RandomState(seed + ep)
        obs, info = env.reset(seed=seed + ep)
        done = False
        pending_indices.clear()
        run_won = False

        while not done:
            mask = info["action_mask"]
            valid = np.where(mask == 1)[0]
            action = int(rng.choice(valid))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            mgr = env._run_env._mgr
            if done:
                run_won = bool(mgr is not None and mgr.player_won)

        sample_weight = 1.0 if run_won else loss_weight
        for idx in pending_indices:
            weights[idx] = sample_weight

        if (ep + 1) % 100 == 0:
            print(f"Collected {ep + 1}/{n_episodes} episodes, {len(labels)} samples")
            save_partial(ep + 1)

    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_path,
        contexts=np.stack(contexts),
        card_features=np.stack(card_features),
        masks=np.stack(masks),
        labels=np.array(labels, dtype=np.int64),
        weights=np.array(weights, dtype=np.float32),
    )
    if partial_path.exists():
        partial_path.unlink()
    print(f"Saved {len(labels)} samples to {data_path}")


def train_model(
    data_path: Path,
    output_dir: Path,
    loss_weight: float,
    epochs: int,
    lr: float,
    val_fraction: float,
    resume: bool = False,
    checkpoint_every: int = 1,
) -> None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("Training requires torch. Install with: pip install 'sts2-rl-agent[train]'")
        sys.exit(1)

    from sts2_env.gym_env.card_value import (
        SKIP_LABEL,
        build_card_value_net,
        load_training_checkpoint,
        save_card_value_model,
        save_training_checkpoint,
        CardValueConfig,
    )

    data = np.load(data_path)
    contexts = data["contexts"]
    card_features = data["card_features"]
    masks = data["masks"]
    labels = data["labels"]
    weights = data["weights"]

    n = len(labels)
    indices = np.arange(n)
    np.random.RandomState(42).shuffle(indices)
    split = int(n * (1.0 - val_fraction))
    train_idx = indices[:split]
    val_idx = indices[split:]

    def make_tensors(idxs):
        return (
            torch.from_numpy(contexts[idxs]),
            torch.from_numpy(card_features[idxs]),
            torch.from_numpy(masks[idxs]),
            torch.from_numpy(labels[idxs]),
            torch.from_numpy(weights[idxs]),
        )

    train_tensors = make_tensors(train_idx)
    val_tensors = make_tensors(val_idx)

    train_loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=256,
        shuffle=True,
    )

    config = CardValueConfig()
    model = build_card_value_net(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="none")

    best_val_acc = -1.0
    start_epoch = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "training_checkpoint.pt"

    if resume and checkpoint_path.exists():
        ckpt = load_training_checkpoint(checkpoint_path)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_acc = float(ckpt["best_val_acc"])
        print(
            f"Resuming training from epoch {start_epoch}/{epochs} "
            f"(best_val_acc={best_val_acc:.3f})"
        )

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            ctx, cards, mask, label, weight = batch
            optimizer.zero_grad()
            logits = model(ctx, cards, mask)
            loss = criterion(logits, label)
            weighted = (loss * weight).mean()
            weighted.backward()
            optimizer.step()
            train_loss += float(weighted.item()) * len(label)
            preds = logits.argmax(dim=1)
            train_correct += int((preds == label).sum().item())
            train_total += len(label)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            ctx, cards, mask, label, _weight = val_tensors
            for start in range(0, len(label), 256):
                end = min(start + 256, len(label))
                logits = model(ctx[start:end], cards[start:end], mask[start:end])
                preds = logits.argmax(dim=1)
                val_correct += int((preds == label[start:end]).sum().item())
                val_total += end - start

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        print(
            f"Epoch {epoch + 1}/{epochs}  "
            f"loss={train_loss / max(train_total, 1):.4f}  "
            f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}"
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_card_value_model(model, output_dir, config)

        if (epoch + 1) % checkpoint_every == 0:
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                config=config,
            )

    print(f"Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train card-value network")
    parser.add_argument(
        "--collect-episodes", type=int, default=0,
        help="Collect this many episodes before training (0 = skip)",
    )
    parser.add_argument(
        "--data", type=str, default=DEFAULT_DATA_PATH,
        help="Path to episodes.npz dataset",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory for best_model.pt and config.json",
    )
    parser.add_argument(
        "--combat-model", type=str, default=DEFAULT_COMBAT_MODEL,
        help="Combat model for data collection rollouts",
    )
    from sts2_env.characters.all import SUPPORTED_TRAINING_CHARACTERS

    parser.add_argument(
        "--character", type=str, default="Ironclad",
        choices=SUPPORTED_TRAINING_CHARACTERS,
        help="Character for data collection rollouts (default: Ironclad)",
    )
    parser.add_argument(
        "--loss-weight", type=float, default=0.3,
        help="Sample weight for picks from losing runs",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Validation fraction",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Collection RNG seed",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from output-dir/training_checkpoint.pt",
    )
    parser.add_argument(
        "--resume-collection", action="store_true",
        help="Resume data collection from a *.partial.npz checkpoint",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=1,
        help="Save a resumable training checkpoint every N epochs (default: 1)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if args.collect_episodes > 0:
        collect_episodes(
            args.collect_episodes,
            data_path,
            args.combat_model,
            character_id=args.character,
            seed=args.seed,
            loss_weight=args.loss_weight,
            resume=args.resume_collection,
        )

    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        print("Run with --collect-episodes N first.")
        sys.exit(1)

    train_model(
        data_path,
        Path(args.output_dir),
        loss_weight=args.loss_weight,
        epochs=args.epochs,
        lr=args.lr,
        val_fraction=args.val_fraction,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()

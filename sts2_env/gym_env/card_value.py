"""Learned card-value network for card reward selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from sts2_env.cards.base import CardInstance
from sts2_env.core.enums import CardRarity, CardType
from sts2_env.gym_env.noncombat_heuristics import (
    CARD_REWARD_LARGE_DECK_SIZE,
    pick_card_reward_index_rules,
)
from sts2_env.run.run_manager import RunManager

MAX_CARD_OPTIONS = 5
SKIP_LABEL = MAX_CARD_OPTIONS
NUM_CARD_TYPES = 3
NUM_RARITIES = 5
RUN_CONTEXT_SIZE = 10
CARD_FEATURE_SIZE = 1 + NUM_CARD_TYPES + NUM_RARITIES + 1
HIDDEN_SIZE = 128

CARD_TYPES_ORDER = (CardType.POWER, CardType.ATTACK, CardType.SKILL)
RARITIES_ORDER = (
    CardRarity.COMMON,
    CardRarity.UNCOMMON,
    CardRarity.RARE,
    CardRarity.BASIC,
    CardRarity.ANCIENT,
)


@dataclass(frozen=True)
class CardValueConfig:
    run_context_size: int = RUN_CONTEXT_SIZE
    card_feature_size: int = CARD_FEATURE_SIZE
    max_card_options: int = MAX_CARD_OPTIONS
    skip_label: int = SKIP_LABEL
    hidden_size: int = HIDDEN_SIZE
    skip_threshold: float = -2.0
    large_deck_skip_size: int = CARD_REWARD_LARGE_DECK_SIZE


def _one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        vec[index] = 1.0
    return vec


def encode_card_features(card: CardInstance) -> np.ndarray:
    """Encode a single offered card (~10 dims)."""
    features = np.zeros(CARD_FEATURE_SIZE, dtype=np.float32)
    features[0] = min(card.cost, 5) / 5.0
    type_index = next(
        (i for i, card_type in enumerate(CARD_TYPES_ORDER) if card.card_type == card_type),
        -1,
    )
    if type_index >= 0:
        features[1 + type_index] = 1.0
    rarity_index = next(
        (i for i, rarity in enumerate(RARITIES_ORDER) if card.rarity == rarity),
        -1,
    )
    if rarity_index >= 0:
        features[1 + NUM_CARD_TYPES + rarity_index] = 1.0
    features[-1] = 1.0 if card.upgraded else 0.0
    return features


def encode_run_context(mgr: RunManager) -> np.ndarray:
    """Encode run state at a card reward screen (~10 dims)."""
    rs = mgr.run_state
    player = rs.player
    deck = player.deck
    context = np.zeros(RUN_CONTEXT_SIZE, dtype=np.float32)
    context[0] = player.current_hp / max(player.max_hp, 1)
    context[1] = len(deck) / 40.0
    context[2] = sum(1 for c in deck if c.card_type == CardType.ATTACK) / max(len(deck), 1)
    context[3] = sum(1 for c in deck if c.card_type == CardType.SKILL) / max(len(deck), 1)
    context[4] = sum(1 for c in deck if c.card_type == CardType.POWER) / max(len(deck), 1)
    context[5] = rs.current_act_index / 3.0
    context[6] = rs.total_floor / 50.0
    context[7] = player.gold / 1000.0
    context[8] = len(rs.relics) / 30.0
    context[9] = 1.0 if any(a.get("action") == "skip" for a in mgr.get_available_actions()) else 0.0
    return context


def encode_card_reward_sample(mgr: RunManager) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (context, card_features, mask, num_cards) for the current reward screen."""
    cards = list(mgr._offered_cards)
    context = encode_run_context(mgr)
    card_features = np.zeros((MAX_CARD_OPTIONS, CARD_FEATURE_SIZE), dtype=np.float32)
    mask = np.zeros(MAX_CARD_OPTIONS, dtype=np.float32)
    for i, card in enumerate(cards[:MAX_CARD_OPTIONS]):
        card_features[i] = encode_card_features(card)
        mask[i] = 1.0
    return context, card_features, mask, len(cards)


def label_from_rules(mgr: RunManager) -> int:
    """Heuristic label index, or SKIP_LABEL."""
    pick = pick_card_reward_index_rules(mgr)
    if pick is None:
        return SKIP_LABEL
    return int(pick)


def build_card_value_net(config: CardValueConfig | None = None):
    """Construct an untrained CardValueNet (requires torch)."""
    import torch.nn as nn

    cfg = config or CardValueConfig()
    num_actions = cfg.max_card_options + 1

    class CardValueNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = cfg
            self.context_net = nn.Sequential(
                nn.Linear(cfg.run_context_size, cfg.hidden_size),
                nn.ReLU(),
            )
            self.card_net = nn.Sequential(
                nn.Linear(cfg.card_feature_size, cfg.hidden_size),
                nn.ReLU(),
            )
            self.card_head = nn.Linear(cfg.hidden_size * 2, 1)
            self.skip_head = nn.Linear(cfg.hidden_size, 1)

        def forward(
            self,
            context: Any,
            card_features: Any,
            mask: Any,
        ) -> Any:
            import torch

            batch = context.shape[0]
            ctx_emb = self.context_net(context)
            card_emb = self.card_net(card_features)
            ctx_expanded = ctx_emb.unsqueeze(1).expand(-1, cfg.max_card_options, -1)
            combined = torch.cat([ctx_expanded, card_emb], dim=-1)
            card_logits = self.card_head(combined).squeeze(-1)
            skip_logit = self.skip_head(ctx_emb)
            logits = torch.cat([card_logits, skip_logit.unsqueeze(1)], dim=1)
            full_mask = torch.ones(batch, num_actions, device=context.device)
            full_mask[:, : cfg.max_card_options] = mask
            full_mask[:, cfg.skip_label] = 1.0
            return logits.masked_fill(full_mask == 0, -1e8)

    return CardValueNet()


def pick_card_reward_index(
    mgr: RunManager,
    model: Any,
    config: CardValueConfig | None = None,
) -> int | None:
    """Pick card index using the learned model, or None to skip."""
    cfg = config or CardValueConfig()
    cards = list(mgr._offered_cards)
    actions = mgr.get_available_actions()
    can_skip = any(a.get("action") == "skip" for a in actions)
    if not cards:
        return None if can_skip else 0
    if can_skip and len(mgr.run_state.player.deck) > cfg.large_deck_skip_size:
        return None

    import torch

    context, card_features, mask, _ = encode_card_reward_sample(mgr)
    with torch.no_grad():
        logits = model(
            torch.from_numpy(context).unsqueeze(0),
            torch.from_numpy(card_features).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        )
    logits_np = logits.squeeze(0).cpu().numpy()
    if can_skip and logits_np[cfg.skip_label] >= cfg.skip_threshold:
        if logits_np[cfg.skip_label] >= float(np.max(logits_np[: len(cards)])):
            return None
    pick = int(np.argmax(logits_np[: len(cards)]))
    return pick


def save_card_value_model(
    model: Any,
    path: str | Path,
    config: CardValueConfig | None = None,
) -> None:
    """Save weights and config.json."""
    import torch

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = config or CardValueConfig()
    if out.suffix == ".pt":
        model_path = out
        config_path = out.parent / "config.json"
    else:
        out.mkdir(parents=True, exist_ok=True)
        model_path = out / "best_model.pt"
        config_path = out / "config.json"
    torch.save(model.state_dict(), model_path)
    config_path.write_text(
        json.dumps(
            {
                "run_context_size": cfg.run_context_size,
                "card_feature_size": cfg.card_feature_size,
                "max_card_options": cfg.max_card_options,
                "skip_label": cfg.skip_label,
                "hidden_size": cfg.hidden_size,
                "skip_threshold": cfg.skip_threshold,
                "large_deck_skip_size": cfg.large_deck_skip_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_card_value_model(path: str | Path) -> tuple[Any, CardValueConfig]:
    """Load CardValueNet and config from a .pt file or output directory."""
    import torch

    model_path = Path(path)
    if model_path.is_dir():
        config_path = model_path / "config.json"
        weights_path = model_path / "best_model.pt"
    else:
        config_path = model_path.parent / "config.json"
        weights_path = model_path
    cfg_dict = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = CardValueConfig(**cfg_dict)
    model = build_card_value_net(cfg)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, cfg

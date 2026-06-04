"""Tests for card-value training utilities."""

import numpy as np
import pytest

from sts2_env.gym_env.card_value import (
    CARD_FEATURE_SIZE,
    MAX_CARD_OPTIONS,
    RUN_CONTEXT_SIZE,
    SKIP_LABEL,
    build_card_value_net,
    save_card_value_model,
    load_card_value_model,
    CardValueConfig,
)


def test_weighted_loss_decreases(tmp_path):
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    n = 64
    contexts = np.random.randn(n, RUN_CONTEXT_SIZE).astype(np.float32)
    cards = np.random.randn(n, MAX_CARD_OPTIONS, CARD_FEATURE_SIZE).astype(np.float32)
    masks = np.ones((n, MAX_CARD_OPTIONS), dtype=np.float32)
    labels = np.random.randint(0, SKIP_LABEL + 1, size=n)
    weights = np.where(np.random.rand(n) > 0.5, 1.0, 0.3).astype(np.float32)

    config = CardValueConfig(hidden_size=32)
    model = build_card_value_net(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss(reduction="none")

    ctx = torch.from_numpy(contexts)
    card_t = torch.from_numpy(cards)
    mask_t = torch.from_numpy(masks)
    label_t = torch.from_numpy(labels)
    weight_t = torch.from_numpy(weights)

    loss_start = None
    loss_end = None
    for step in range(30):
        optimizer.zero_grad()
        logits = model(ctx, card_t, mask_t)
        loss = (criterion(logits, label_t) * weight_t).mean()
        if step == 0:
            loss_start = float(loss.item())
        loss_end = float(loss.item())
        loss.backward()
        optimizer.step()

    assert loss_end < loss_start


def test_save_load_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")

    config = CardValueConfig(hidden_size=32)
    model = build_card_value_net(config)
    out = tmp_path / "card_value"
    save_card_value_model(model, out, config)
    loaded, loaded_cfg = load_card_value_model(out)
    assert loaded_cfg.hidden_size == 32

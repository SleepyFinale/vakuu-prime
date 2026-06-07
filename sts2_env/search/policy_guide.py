"""PPO policy prior and critic value for MCTS."""

from __future__ import annotations

from typing import Any

import numpy as np

from sts2_env.core.constants import ACTION_SPACE_SIZE
from sts2_env.gym_env.combat_value import predict_combat_values


def policy_prior_and_value(
    model: Any,
    obs: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return masked action prior probabilities and critic value V(s)."""
    import torch

    obs_arr = np.asarray(obs, dtype=np.float32)
    mask_arr = np.asarray(mask, dtype=np.int8)
    obs_tensor = torch.as_tensor(obs_arr, device=model.device).unsqueeze(0)
    mask_tensor = torch.as_tensor(mask_arr, device=model.device).unsqueeze(0)

    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor, action_masks=mask_tensor)
        probs = distribution.distribution.probs.detach().cpu().numpy().reshape(-1)
        value = float(predict_combat_values(model, obs_arr)[0])

    priors = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
    priors[: len(probs)] = probs
    legal = mask_arr.astype(bool)
    if legal.any():
        total = priors[legal].sum()
        if total <= 0:
            priors[legal] = 1.0 / legal.sum()
        else:
            priors[~legal] = 0.0
            priors[legal] /= total
    return priors, value

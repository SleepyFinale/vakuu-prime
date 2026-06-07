"""Combat critic value estimation for deck evaluation and draft scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from sts2_env.cards.base import CardInstance, reset_instance_counter
from sts2_env.core.combat import CombatState
from sts2_env.core.rng import Rng
from sts2_env.encounters.pools import encounter_lists_for_act
from sts2_env.encounters.registry import EncounterSetup
from sts2_env.gym_env.card_value import CARD_REWARD_LARGE_DECK_SIZE
from sts2_env.gym_env.observation import encode_observation
from sts2_env.run.run_manager import RunManager

EncounterTier = Literal["weak", "normal", "elite", "boss"]


@dataclass(frozen=True)
class CombatValueConfig:
    encounter_tier: EncounterTier = "elite"
    num_encounters: int = 3
    rng_seed: int = 0
    skip_threshold: float = -0.05
    large_deck_skip_size: int = CARD_REWARD_LARGE_DECK_SIZE


def _get_encounter_pool(mgr: RunManager, tier: EncounterTier) -> list[EncounterSetup]:
    rs = mgr.run_state
    act = rs.current_act
    weak, normal, elite, boss = encounter_lists_for_act(
        rs.current_act_index,
        biome_id=act.biome_id if rs.current_act_index == 0 else None,
    )
    pools: dict[str, list[EncounterSetup]] = {
        "weak": weak,
        "normal": normal,
        "elite": elite,
        "boss": boss,
    }
    return list(pools.get(tier, elite))


def _select_encounters(
    pool: list[EncounterSetup],
    num_encounters: int,
    rng_seed: int,
) -> list[EncounterSetup]:
    if not pool:
        return []
    rng = np.random.default_rng(rng_seed)
    if len(pool) <= num_encounters:
        return list(pool)
    indices = rng.choice(len(pool), size=num_encounters, replace=False)
    return [pool[i] for i in sorted(indices)]


def clone_run_deck(mgr: RunManager, extra_card: CardInstance | None = None) -> list[CardInstance]:
    """Clone the current run deck, optionally appending one card."""
    player = mgr.run_state.player
    deck = [player.clone_card_for_deck(card) for card in player.deck]
    if extra_card is not None:
        deck.append(player.clone_card_for_deck(extra_card))
    return deck


def build_combat_from_deck(
    mgr: RunManager,
    deck: list[CardInstance],
    encounter_setup: EncounterSetup,
    rng_seed: int,
) -> CombatState:
    """Bootstrap a combat encounter from a hypothetical deck (does not mutate the run)."""
    rs = mgr.run_state
    player = rs.player
    reset_instance_counter()
    combat = CombatState(
        player_hp=player.current_hp,
        player_max_hp=player.max_hp,
        deck=list(deck),
        rng_seed=rng_seed,
        relics=list(rs.relics),
        gold=player.gold,
        character_id=player.character_id,
        potions=list(player.potions),
        max_potion_slots=player.max_potion_slots,
        ascension_level=rs.ascension_level,
    )
    encounter_rng = Rng(rng_seed + 1)
    encounter_setup(combat, encounter_rng)
    combat.start_combat()
    return combat


def predict_combat_values(
    combat_model: Any,
    observations: np.ndarray,
) -> np.ndarray:
    """Read critic values V(s) for one or more combat observations."""
    import torch

    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[np.newaxis, :]
    obs_tensor = torch.as_tensor(obs, device=combat_model.device)
    with torch.no_grad():
        values = combat_model.policy.predict_values(obs_tensor)
    return values.detach().cpu().numpy().reshape(-1)


def estimate_encounter_value(
    mgr: RunManager,
    deck: list[CardInstance],
    combat_model: Any,
    encounter_setup: EncounterSetup,
    rng_seed: int,
) -> float:
    """Return V(s0) for a single encounter bootstrap."""
    combat = build_combat_from_deck(mgr, deck, encounter_setup, rng_seed)
    obs = encode_observation(combat)
    return float(predict_combat_values(combat_model, obs)[0])


def estimate_deck_value(
    mgr: RunManager,
    deck: list[CardInstance],
    combat_model: Any,
    *,
    config: CombatValueConfig | None = None,
    encounter_tier: EncounterTier | None = None,
    num_encounters: int | None = None,
    rng_seed: int | None = None,
) -> float:
    """Average combat critic value over sampled encounters for a deck."""
    cfg = config or CombatValueConfig()
    tier = encounter_tier or cfg.encounter_tier
    n_enc = num_encounters if num_encounters is not None else cfg.num_encounters
    seed = rng_seed if rng_seed is not None else cfg.rng_seed

    pool = _get_encounter_pool(mgr, tier)
    setups = _select_encounters(pool, n_enc, seed)
    if not setups:
        return 0.0

    values = []
    for i, setup in enumerate(setups):
        values.append(
            estimate_encounter_value(mgr, deck, combat_model, setup, seed + i * 9973)
        )
    return float(np.mean(values))


def score_card_draft_options(
    mgr: RunManager,
    combat_model: Any,
    *,
    config: CombatValueConfig | None = None,
) -> tuple[list[float], float]:
    """Return per-card delta-V scores and baseline deck value."""
    cfg = config or CombatValueConfig()
    cards = list(mgr._offered_cards)
    baseline_deck = clone_run_deck(mgr)
    baseline = estimate_deck_value(mgr, baseline_deck, combat_model, config=cfg)
    deltas: list[float] = []
    for card in cards:
        deck_with = clone_run_deck(mgr, extra_card=card)
        value = estimate_deck_value(mgr, deck_with, combat_model, config=cfg)
        deltas.append(value - baseline)
    return deltas, baseline


def pick_card_by_combat_value(
    mgr: RunManager,
    combat_model: Any,
    *,
    config: CombatValueConfig | None = None,
) -> tuple[int | None, list[float], float]:
    """Pick card index by max delta-V, or None to skip."""
    cfg = config or CombatValueConfig()
    cards = list(mgr._offered_cards)
    actions = mgr.get_available_actions()
    can_skip = any(a.get("action") == "skip" for a in actions)
    if not cards:
        return (None if can_skip else 0), [], 0.0
    if can_skip and len(mgr.run_state.player.deck) > cfg.large_deck_skip_size:
        return None, [], 0.0

    deltas, baseline = score_card_draft_options(mgr, combat_model, config=cfg)
    if not deltas:
        return 0, deltas, baseline

    best_index = int(np.argmax(deltas))
    best_delta = deltas[best_index]
    if can_skip and best_delta < cfg.skip_threshold:
        if all(d < cfg.skip_threshold for d in deltas):
            return None, deltas, baseline
    return best_index, deltas, baseline


def draft_value_for_pick(
    mgr: RunManager,
    pick_index: int | None,
    combat_model: Any,
    *,
    config: CombatValueConfig | None = None,
) -> float:
    """Delta-V for a chosen card index (None / skip => 0)."""
    if pick_index is None:
        return 0.0
    cards = list(mgr._offered_cards)
    if pick_index < 0 or pick_index >= len(cards):
        return 0.0
    cfg = config or CombatValueConfig()
    baseline_deck = clone_run_deck(mgr)
    baseline = estimate_deck_value(mgr, baseline_deck, combat_model, config=cfg)
    deck_with = clone_run_deck(mgr, extra_card=cards[pick_index])
    value = estimate_deck_value(mgr, deck_with, combat_model, config=cfg)
    return value - baseline

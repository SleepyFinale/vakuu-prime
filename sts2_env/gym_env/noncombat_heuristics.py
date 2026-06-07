"""Rule-based non-combat decisions for full-run RL training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from sts2_env.core.enums import CardType
from sts2_env.gym_env.run_env import _ActionLayout, _LAYOUT
from sts2_env.run.run_manager import RunManager

CardRewardPicker = Callable[[RunManager], int | None]

CARD_REWARD_LARGE_DECK_SIZE = 30
REST_HP_RATIO_THRESHOLD = 0.5
CARD_REWARD_TYPE_PRIORITY = (CardType.POWER, CardType.ATTACK, CardType.SKILL)
BOSS_RELIC_PREFERENCES = ("BLACK_STAR", "ASTROLABE", "CALLING_BELL", "SNECKO_EYE")


@dataclass
class NoncombatHeuristicConfig:
    card_reward: bool = True
    boss_relic: bool = True
    rest_site: bool = True
    card_reward_mode: Literal["rules", "learned", "combat_value"] = "rules"
    card_value_model: Any | None = None
    card_value_config: Any | None = None
    combat_value_model: Any | None = None
    combat_value_config: Any | None = None


def _pick_card_with_config(mgr: RunManager, config: NoncombatHeuristicConfig) -> int | None:
    if config.card_reward_mode == "combat_value" and config.combat_value_model is not None:
        from sts2_env.gym_env.combat_value import pick_card_by_combat_value

        pick, _, _ = pick_card_by_combat_value(
            mgr,
            config.combat_value_model,
            config=config.combat_value_config,
        )
        return pick
    if config.card_reward_mode == "learned" and config.card_value_model is not None:
        from sts2_env.gym_env.card_value import pick_card_reward_index as pick_learned

        return pick_learned(mgr, config.card_value_model, config.card_value_config)
    return pick_card_reward_index_rules(mgr)


def _hp_ratio(mgr: RunManager) -> float:
    player = mgr.run_state.player
    return player.current_hp / max(player.max_hp, 1)


def pick_card_reward_index_rules(mgr: RunManager) -> int | None:
    """Pick card index using rules, or None to skip."""
    cards = list(mgr._offered_cards)
    actions = mgr.get_available_actions()
    can_skip = any(a.get("action") == "skip" for a in actions)
    if not cards:
        return None if can_skip else 0
    if can_skip and len(mgr.run_state.player.deck) > CARD_REWARD_LARGE_DECK_SIZE:
        return None
    for card_type in CARD_REWARD_TYPE_PRIORITY:
        for index, card in enumerate(cards):
            if card.card_type == card_type:
                return index
    return 0


def pick_card_reward_index(
    mgr: RunManager,
    config: NoncombatHeuristicConfig | None = None,
) -> int | None:
    """Pick card index using config (rules or learned model)."""
    if config is None:
        return pick_card_reward_index_rules(mgr)
    return _pick_card_with_config(mgr, config)


def pick_boss_relic_index(mgr: RunManager) -> int:
    relics = list(mgr._boss_relics)
    if not relics:
        return 0
    for preferred in BOSS_RELIC_PREFERENCES:
        for index, relic_id in enumerate(relics):
            if relic_id == preferred:
                return index
    return 0


def pick_rest_action_index(mgr: RunManager) -> int:
    rest_actions = [
        a for a in mgr.get_available_actions()
        if a.get("action") == "rest_option"
    ]
    if not rest_actions:
        return 0
    hp_ratio = _hp_ratio(mgr)
    preferred = "HEAL" if hp_ratio < REST_HP_RATIO_THRESHOLD else "SMITH"
    for index, action in enumerate(rest_actions):
        option_id = str(action.get("option_id", "")).upper()
        if option_id == preferred:
            return index
    if preferred == "SMITH":
        for index, action in enumerate(rest_actions):
            if str(action.get("option_id", "")).upper() == "HEAL":
                return index
    return 0


def card_reward_global_action(
    mgr: RunManager,
    layout: _ActionLayout = _LAYOUT,
    config: NoncombatHeuristicConfig | None = None,
) -> int | None:
    """Return unified action index for CARD_REWARD, or None if not applicable."""
    if mgr.phase != RunManager.PHASE_CARD_REWARD:
        return None
    actions = mgr.get_available_actions()
    if any(a.get("action") == "pick_potion" for a in actions):
        return layout.card_reward_start
    if any(a.get("action") == "pick_relic_reward" for a in actions):
        return layout.card_reward_start
    pick_index = pick_card_reward_index(mgr, config)
    if pick_index is None:
        return layout.card_reward_start + 3
    if pick_index < 3:
        return layout.card_reward_start + pick_index
    if pick_index < 3 + layout.card_reward_extra_size:
        return layout.card_reward_extra_start + (pick_index - 3)
    return layout.card_reward_start + 3


def boss_relic_global_action(
    mgr: RunManager,
    layout: _ActionLayout = _LAYOUT,
) -> int | None:
    if mgr.phase != RunManager.PHASE_BOSS_RELIC:
        return None
    index = pick_boss_relic_index(mgr)
    return layout.boss_relic_start + min(index, layout.boss_relic_size - 1)


def rest_site_global_action(
    mgr: RunManager,
    layout: _ActionLayout = _LAYOUT,
) -> int | None:
    if mgr.phase != RunManager.PHASE_REST_SITE:
        return None
    index = pick_rest_action_index(mgr)
    return layout.rest_start + min(index, layout.rest_size - 1)


def heuristic_global_action(
    mgr: RunManager,
    config: NoncombatHeuristicConfig,
    layout: _ActionLayout = _LAYOUT,
) -> int | None:
    """Return a unified action for the current phase, if heuristics apply."""
    if config.card_reward:
        action = card_reward_global_action(mgr, layout, config)
        if action is not None:
            return action
    if config.boss_relic:
        action = boss_relic_global_action(mgr, layout)
        if action is not None:
            return action
    if config.rest_site:
        action = rest_site_global_action(mgr, layout)
        if action is not None:
            return action
    return None


def should_auto_resolve_phase(
    mgr: RunManager,
    config: NoncombatHeuristicConfig,
) -> bool:
    """True when the current phase should be handled by heuristics."""
    if mgr.is_over:
        return False
    phase = mgr.phase
    if config.card_reward and phase == RunManager.PHASE_CARD_REWARD:
        return True
    if config.boss_relic and phase == RunManager.PHASE_BOSS_RELIC:
        return True
    if config.rest_site and phase == RunManager.PHASE_REST_SITE:
        return True
    return False

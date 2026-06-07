"""State adapter: converts game JSON state to observation vectors.

Translates the JSON state received from the C# bridge mod into the
same flat float32 observation vector format used by the gym_env. This
ensures the trained model receives inputs in the exact same encoding
it was trained on.

The observation format is defined in gym_env/observation.py:
  - Player state: hp/max_hp, block/50, energy/10, max_energy/10, ascension/20, turn_count/20 (6)
  - Player powers: all PowerId values (268, amount/20)
  - Hand cards: card_id_norm, cost, damage, block, is_attack, is_power,
                has_exhaust, has_retain, hit_count (10 * 9 = 90)
  - Pile sizes: draw, discard, exhaust counts + pile memory (31) + reserved (3) (37)
  - Enemies: alive, hp%, block, intent_onehot(5), intent_dmg/60,
             intent_hits/min(hits,10)/10, all powers (268) (5 * 278 = 1390)
  - Character mechanics: one-hot(5), stars, orb cap/count, orbs(3*2), osty(3) (17)
  - Relics: relic_id_norm, rarity, enabled, is_used_up, counter_norm (30 * 5 = 150)
  - Potions: potion_id_norm, rarity, can_use_in_combat (9 * 3 = 27)
  Total: 1985 dimensions
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sts2_env.core.constants import (
    ACTION_END_TURN,
    ACTION_SPACE_SIZE,
    MAX_ENEMIES,
    MAX_HAND_SIZE,
    MAX_POTION_SLOTS,
    POTION_ACTION_START,
    POTION_TARGET_OPTIONS,
)
from sts2_env.core.enums import PotionUsageType
from sts2_env.gym_env.pile_distribution import (
    PILE_FEATURES,
    cards_from_bridge,
    encode_pile_summaries,
    projected_next_draw_count,
)
from sts2_env.gym_env.observation import (
    CARD_FEATURES,
    ENEMY_CORE_FEATURES,
    ENEMY_FEATURES,
    ENEMY_POWERS,
    OBS_SIZE,
    COMBAT_OBS_V2_SIZE,
    OBS_ASCENSION_SCALE,
    OBS_TURN_COUNT_CAP,
    PLAYER_CORE_FEATURES,
    PLAYER_POWERS,
    RELIC_OBS_SIZE,
    encode_card_features_from_fields,
    encode_character_mechanics_from_fields,
    encode_potions_into_obs,
    encode_relics_into_obs,
    intent_types_from_names,
    write_enemy_intent_features,
    write_empty_hand_slot,
)
from sts2_env.bridge.protocol import (
    Phase,
    TargetTypeName,
)


def _bridge_intent_type_names(enemy: dict[str, Any]) -> list[str]:
    raw = enemy.get("intent_types")
    if raw:
        return [str(name) for name in raw]
    intent = enemy.get("intent")
    if intent:
        return [str(intent)]
    return []


_TRACKED_PLAYER_POWERS = [p.name for p in PLAYER_POWERS]
_TRACKED_ENEMY_POWERS = [p.name for p in ENEMY_POWERS]

# Target types that need specific enemy targeting (for action masking)
_TARGETED_TYPES = {TargetTypeName.ANY_ENEMY, "ANY_ENEMY", "RANDOM_ENEMY", TargetTypeName.RANDOM_ENEMY}
_UNTARGETED_TYPES = {TargetTypeName.SELF, TargetTypeName.NONE, TargetTypeName.ALL_ENEMIES,
                     "SELF", "NONE", "ALL_ENEMIES", "Self", "None", "AllEnemies"}
_POTION_TARGETED_TYPES = {TargetTypeName.ANY_ENEMY, "ANY_ENEMY", "AnyEnemy"}
_POTION_UNTARGETED_TYPES = {
    TargetTypeName.SELF,
    TargetTypeName.ALL_ENEMIES,
    "SELF",
    "ANY_PLAYER",
    "ALL_ENEMIES",
    "Self",
    "AnyPlayer",
    "AllEnemies",
}


class StateAdapter:
    """Converts game state JSON to observation vectors and action masks.

    This adapter bridges the gap between the C# serializer's JSON format
    and the gym environment's numpy observation encoding.

    Usage::

        adapter = StateAdapter()
        state = client.receive_state()
        obs = adapter.encode_observation(state)
        mask = adapter.compute_action_mask(state)
    """

    def encode_observation(self, state: dict[str, Any]) -> np.ndarray:
        """Convert a game state JSON dict to a flat float32 observation vector.

        Args:
            state: Full game state dict as received from the bridge.
                   Must contain 'combat_state' with player, hand, enemies.

        Returns:
            Float32 numpy array of shape (OBS_SIZE,) = (1985,).
            Returns zeros if not in combat.
        """
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

        # Support both formats:
        # Legacy nested payload: state["combat_state"]["player"]
        # Current v2 payload: state["player"]
        combat = state.get("combat_state") or state
        if "player" not in combat:
            return obs

        player = combat.get("player", {})
        # Debug: print first call to verify data
        if not hasattr(self, '_logged_first'):
            self._logged_first = True
            import logging
            logging.getLogger(__name__).warning(
                "First encode_observation: player=%s, hand_count=%d, enemies_count=%d",
                player, len(combat.get("hand", [])), len(combat.get("enemies", []))
            )

        idx = 0

        # --- Player state (6) ---
        player = combat.get("player", {})
        max_hp = player.get("max_hp", 1)
        obs[idx] = player.get("hp", 0) / max(max_hp, 1)
        obs[idx + 1] = player.get("block", 0) / 50.0
        obs[idx + 2] = player.get("energy", 0) / 10.0
        obs[idx + 3] = player.get("max_energy", 3) / 10.0
        ascension = int(combat.get("ascension_level", state.get("ascension_level", 0)))
        obs[idx + 4] = ascension / OBS_ASCENSION_SCALE
        turn = max(1, int(combat.get("round", 1)))
        obs[idx + 5] = min(turn, OBS_TURN_COUNT_CAP) / OBS_TURN_COUNT_CAP
        idx += PLAYER_CORE_FEATURES

        # --- Player powers (268) ---
        player_powers = _powers_to_dict(player.get("powers", []))
        for power_name in _TRACKED_PLAYER_POWERS:
            obs[idx] = player_powers.get(power_name, 0) / 20.0
            idx += 1

        # --- Hand cards (10 * 9 = 90) ---
        hand = combat.get("hand", [])
        for i in range(MAX_HAND_SIZE):
            if i < len(hand):
                card = hand[i]
                obs[idx:idx + CARD_FEATURES] = encode_card_features_from_fields(
                    card_id=str(card.get("id", "")),
                    cost=int(card.get("cost", 0)),
                    card_type=str(card.get("type", "")),
                    base_damage=card.get("base_damage"),
                    base_block=card.get("base_block"),
                    keywords=card.get("keywords"),
                    retain=bool(card.get("retain", False)),
                    single_turn_retain=bool(card.get("single_turn_retain", False)),
                    hit_count=card.get("hit_count"),
                    upgraded=bool(card.get("upgraded", False)),
                )
            else:
                write_empty_hand_slot(obs, idx)
            idx += CARD_FEATURES

        # --- Pile summaries (32) ---
        draw, discard, play, hand_cards = cards_from_bridge(combat)
        exhaust_count = combat.get("exhaust_pile_count", 0)
        next_draw_count = projected_next_draw_count(len(hand))
        obs[idx:idx + PILE_FEATURES] = encode_pile_summaries(
            draw,
            discard,
            play,
            hand_cards,
            exhaust_count,
            next_draw_count=next_draw_count,
        )
        idx += PILE_FEATURES

        # --- Enemies (5 * 278 = 1390) ---
        enemies = combat.get("enemies", [])
        for i in range(MAX_ENEMIES):
            if i < len(enemies):
                enemy = enemies[i]
                is_alive = enemy.get("is_alive", False)
                enemy_max_hp = max(enemy.get("max_hp", 1), 1)

                obs[idx] = 1.0 if is_alive else 0.0
                obs[idx + 1] = enemy.get("hp", 0) / enemy_max_hp
                obs[idx + 2] = enemy.get("block", 0) / 50.0

                # Intent encoding (multi-bit one-hot + aggregated damage + hits)
                if is_alive:
                    intent_types = intent_types_from_names(_bridge_intent_type_names(enemy))
                    write_enemy_intent_features(
                        obs,
                        idx + 3,
                        intent_types=intent_types,
                        total_damage=int(enemy.get("intent_damage", 0) or 0),
                        total_hits=int(enemy.get("intent_hits", 1) or 1),
                    )

                enemy_powers = _powers_to_dict(enemy.get("powers", []))
                power_base = idx + ENEMY_CORE_FEATURES
                for j, power_name in enumerate(_TRACKED_ENEMY_POWERS):
                    obs[power_base + j] = enemy_powers.get(power_name, 0) / 10.0

            idx += ENEMY_FEATURES  # advance even for empty enemy slots

        idx = encode_character_mechanics_from_fields(
            obs,
            idx,
            character_id=player.get("character_id"),
            stars=int(player.get("stars", 0)),
            orb_capacity=int((player.get("orb_queue") or {}).get("capacity", 0)),
            orb_count=int((player.get("orb_queue") or {}).get("count", 0)),
            orbs=_bridge_orb_entries(player.get("orb_queue")),
            osty_alive=bool((player.get("osty") or {}).get("alive", False)),
            osty_hp=int((player.get("osty") or {}).get("hp", 0)),
            osty_max_hp=int((player.get("osty") or {}).get("max_hp", 0)),
            osty_block=int((player.get("osty") or {}).get("block", 0)),
        )
        relics = combat.get("relics") or player.get("relics")
        encode_relics_into_obs(obs, COMBAT_OBS_V2_SIZE, relics)
        potion_start = COMBAT_OBS_V2_SIZE + RELIC_OBS_SIZE
        potions = combat.get("potions") or state.get("run_state", {}).get("potions", [])
        encode_potions_into_obs(obs, potion_start, potions)

        return obs

    def compute_action_mask(self, state: dict[str, Any]) -> np.ndarray:
        """Compute a boolean mask of valid actions from the game state.

        The combat action space is fixed-width and includes cards, end turn,
        and potion uses.

        Card actions:
          - 0: END_TURN
          - 1..10: Play card i (untargeted: self/none/all_enemies)
          - 11..60: Play card i targeting enemy j (i*5 + j offset)

        Potion actions:
          - POTION_ACTION_START..: slot-major layout
          - each slot gets 1 untargeted/self action + MAX_ENEMIES targeted actions

        This matches get_action_mask() in gym_env/action_space.py.

        Args:
            state: Full game state dict from the bridge.

        Returns:
            Int8 numpy array of shape (ACTION_SPACE_SIZE,).
        """
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)

        # Support both formats (combat_state wrapper or flat)
        combat = state.get("combat_state") or state
        if "player" not in combat:
            mask[ACTION_END_TURN] = 1
            return mask

        # Can always end turn during combat
        mask[ACTION_END_TURN] = 1

        player = combat.get("player", {})
        energy = player.get("energy", 0)
        hand = combat.get("hand", [])
        enemies = combat.get("enemies", [])
        available_actions = {
            str(item).upper()
            for item in (combat.get("available_actions") or state.get("available_actions") or [])
        }

        # Build list of alive enemy indices
        alive_enemies = []
        for j in range(min(len(enemies), MAX_ENEMIES)):
            if enemies[j].get("is_alive", False):
                alive_enemies.append(j)

        # For each card in hand, determine valid actions
        for i in range(min(len(hand), MAX_HAND_SIZE)):
            card = hand[i]
            cost = card.get("cost", 0)

            # Check if card is playable (enough energy, cost >= 0)
            if cost < 0:
                # X-cost cards (cost = -1) are always playable if energy > 0
                if energy <= 0:
                    continue
            elif cost > energy:
                continue

            target_type = card.get("target", "Self")

            if target_type in _UNTARGETED_TYPES:
                # Self-target / no-target / all-enemies: action index = 1 + i
                mask[1 + i] = 1
            elif target_type in _TARGETED_TYPES:
                # Needs specific enemy target: action index = 1 + MAX_HAND_SIZE + i * MAX_ENEMIES + j
                for j in alive_enemies:
                    mask[1 + MAX_HAND_SIZE + i * MAX_ENEMIES + j] = 1

        if not available_actions or "POTION" in available_actions:
            potions = combat.get("potions") or state.get("run_state", {}).get("potions", [])
            for list_index, potion in enumerate(potions[:MAX_POTION_SLOTS]):
                if not potion or not potion.get("can_use", True):
                    continue
                usage = str(potion.get("usage", "")).upper()
                if usage == PotionUsageType.AUTOMATIC.name:
                    continue
                slot = int(potion.get("slot", list_index))
                if slot < 0 or slot >= MAX_POTION_SLOTS:
                    continue
                action_base = POTION_ACTION_START + slot * POTION_TARGET_OPTIONS
                target_type = potion.get("target") or potion.get("target_type", "Self")
                requires_target = potion.get("requires_target", False)
                if requires_target or target_type in _POTION_TARGETED_TYPES:
                    for j in alive_enemies:
                        mask[action_base + 1 + j] = 1
                elif target_type in _POTION_UNTARGETED_TYPES:
                    mask[action_base] = 1

        return mask

    def decode_action(
        self, action: int, state: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert an action index to an action command dict.

        Args:
            action: Action index from model.predict().
            state: Current game state (for reference if needed).

        Returns:
            Action dict ready to send via client.send_action().
        """
        if action == ACTION_END_TURN:
            return {"type": "END_TURN"}

        if action >= POTION_ACTION_START:
            adjusted = action - POTION_ACTION_START
            slot = adjusted // POTION_TARGET_OPTIONS
            target_offset = adjusted % POTION_TARGET_OPTIONS
            target_index = target_offset - 1 if target_offset > 0 else -1
            return {
                "type": "PLAY",
                "out_of_hand": True,
                "potion_slot": slot,
                "target_index": target_index,
            }

        if action <= MAX_HAND_SIZE:
            # Untargeted card play
            card_index = action - 1
            return {"type": "PLAY", "card_index": card_index, "target_index": -1}

        # Targeted card play
        adjusted = action - 1 - MAX_HAND_SIZE
        card_index = adjusted // MAX_ENEMIES
        target_index = adjusted % MAX_ENEMIES
        return {
            "type": "PLAY",
            "card_index": card_index,
            "target_index": target_index,
        }


def _powers_to_dict(powers: list[dict[str, Any]]) -> dict[str, int]:
    """Convert a list of power dicts to a {id: amount} mapping."""
    result: dict[str, int] = {}
    for p in powers:
        pid = p.get("id", "")
        amount = p.get("amount", 0)
        # Normalise to uppercase for matching
        result[pid.upper()] = amount
    return result


def _bridge_orb_entries(orb_queue: dict[str, Any] | None) -> list[tuple[str, int]]:
    if not orb_queue:
        return []
    entries: list[tuple[str, int]] = []
    for orb in orb_queue.get("orbs", [])[:3]:
        if not isinstance(orb, dict):
            continue
        entries.append((
            str(orb.get("type", "UNKNOWN")),
            int(orb.get("evoke_value", 0)),
        ))
    return entries
